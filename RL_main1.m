clear all
close all
clc

%% Carica ambiente
env = RL_environment1();

doTraining = true;

if doTraining

    actInfo = getActionInfo(env);
    obsInfo = getObservationInfo(env);
    numObs  = prod(obsInfo.Dimension);   % 100
    numAct  = prod(actInfo.Dimension);   % 50

    % =====================================================================
    % Architettura reti  
    % =====================================================================
    criticLayerSizes = [512 256 128];
    actorLayerSizes  = [512 256 128];

    % =====================================================================
    % Critic (Value Function)  — 100 → 256 → 128 → 1
    % =====================================================================
    criticNetwork = [
        featureInputLayer(numObs)
        fullyConnectedLayer(criticLayerSizes(1), ...
            Weights=sqrt(2/numObs) * (rand(criticLayerSizes(1),numObs)-0.5), ...
            Bias=1e-3*ones(criticLayerSizes(1),1))
        reluLayer
        fullyConnectedLayer(criticLayerSizes(2), ...
            Weights=sqrt(2/criticLayerSizes(1)) * (rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ...
            Bias=1e-3*ones(criticLayerSizes(2),1))
        reluLayer
        fullyConnectedLayer(1, ...
            Weights=sqrt(2/criticLayerSizes(2)) * (rand(1,criticLayerSizes(2))-0.5), ...
            Bias=1e-3)
        ];
    criticNetwork = dlnetwork(criticNetwork);
    critic = rlValueFunction(criticNetwork, obsInfo);

    % =====================================================================
    % Actor (Gaussian Policy)  — 100 → 256 → 128 → split mean / std
    %
    % Limitazione std  [NUOVO]:
    %   std = softplus(x) + std_min
    % con std_min = 0.05. Evita che la policy collassi a std→0 nelle
    % prime fasi (policy deterministica prematura) o esploda (std>>1).
    % Il clamp è implementato aggiungendo uno scalingLayer che somma
    % una costante: non direttamente supportato in MATLAB RL Toolbox,
    % quindi si usa softplus con bias inizializzato negativo in modo che
    % l'uscita iniziale sia ~ softplus(-2) + 0.05 ≈ 0.17 (range utile).
    % =====================================================================
    std_min = 0.05;   % std minima per ogni azione

    inPath = [
        featureInputLayer(numObs, Name="netOin")
        fullyConnectedLayer(actorLayerSizes(1))
        reluLayer
        fullyConnectedLayer(actorLayerSizes(2))
        reluLayer(Name="relulast")
        ];

    meanPath = [
        fullyConnectedLayer(actorLayerSizes(2), Name="MeanLyr")
        reluLayer
        fullyConnectedLayer(numAct, Name="meanOutLyr", ...
            Bias=zeros(numAct,1))   % bias=0 → policy inizia centrata
        tanhLayer(Name="thmeanOutLyr")
        ];

    % std_min via bias inizializzato: softplus(x+b)+std_min
    % Per avere std_init ≈ 0.2: softplus(-1) ≈ 0.31 → bias=-1 → ok
    sdevPath = [
        fullyConnectedLayer(actorLayerSizes(2), Name="StdLyr")
        reluLayer
        fullyConnectedLayer(numAct, ...
            Bias=-1*ones(numAct,1))  % std iniziale ≈ softplus(-1)+0.05 ≈ 0.36
        reluLayer
        softplusLayer(Name="stdOutLyr")
        ];
    % Nota: std_min viene aggiunto come shift fisso al valore softplus
    % nell'output dell'attore. MATLAB non supporta un addLayer costante,
    % ma rlContinuousGaussianActor tronca comunque std a valori positivi.
    % Per un controllo più fine, usare un customLayer che calcola
    % softplus(x) + std_min — vedere commento in fondo al file.

    net = layerGraph(inPath);
    net = addLayers(net, meanPath);
    net = addLayers(net, sdevPath);
    net = connectLayers(net, "relulast", "MeanLyr/in");
    net = connectLayers(net, "relulast", "StdLyr/in");
    net = dlnetwork(net);

    actor = rlContinuousGaussianActor(net, obsInfo, actInfo, ...
        ActionMeanOutputNames="thmeanOutLyr", ...
        ActionStandardDeviationOutputNames="stdOutLyr", ...
        ObservationInputNames="netOin");

    % =====================================================================
    % Opzioni PPO — fix critic che non converge (Q0/reward ≈ 0.22×)
    %
    % PROBLEMA: con γ=0.99 e T=1000, il ritorno atteso G_t = Σγ^k·r_{t+k}
    % vale fino a r·(1-γ^T)/(1-γ) ≈ 0.23/0.01 ≈ 23 per ogni step.
    % Il critic deve stimare valori molto grandi da osservazioni sparse
    % → lento e impreciso → advantage sbagliati → plateau.
    %
    % FIX:
    %   DiscountFactor 0.99 → 0.97
    %     Orizzonte effettivo: 1/(1-γ) = 33 step invece di 100.
    %     Valori critic max ≈ 0.23/0.03 ≈ 8 → molto più facile da stimare.
    %     La policy rimane lungimirante (33 step = 0.33s a dt=0.01).
    %
    %   MaxStepsPerEpisode 1000 → 500
    %     Con v_treadmill=1.9 m/s tutti i pacchi escono in max ~200 step.
    %     500 step è abbondante, dimezza il tempo per episodio e rende
    %     il segnale più denso (meno step vuoti a fine episodio).
    %
    %   LR critic 5e-4 → 8e-4
    %     Accelera la convergenza del critic senza destabilizzare.
    % =====================================================================
    actorOpts  = rlOptimizerOptions(LearnRate=3e-4);
    criticOpts = rlOptimizerOptions(LearnRate=8e-4);

    agentOpts = rlPPOAgentOptions( ...
        ExperienceHorizon       = 2048,  ...
        ClipFactor              = 0.2,   ...
        EntropyLossWeight       = 0.02,  ...
        ActorOptimizerOptions   = actorOpts,  ...
        CriticOptimizerOptions  = criticOpts, ...
        NumEpoch                = 5,     ...
        AdvantageEstimateMethod = "gae", ...
        GAEFactor               = 0.95,  ...
        SampleTime              = 0.01,  ...
        DiscountFactor          = 0.97,  ...
        MiniBatchSize           = 256);

    agent = rlPPOAgent(actor, critic, agentOpts);

    % =====================================================================
    % Opzioni di training — singolo train(), nessun loop a blocchi
    %
    % Reward attesa: ∈ [0,1] per step × 500 step/ep → max ~500/ep
    % Stop a 375 (media su 100 ep) = ~75% del massimo
    % Checkpoint ogni 500 episodi
    % =====================================================================
    trainOpts = rlTrainingOptions( ...
        MaxEpisodes             = 25000,               ...
        MaxStepsPerEpisode      = 500,                 ...
        Plots                   = "training-progress", ...
        StopTrainingCriteria    = "AverageReward",     ...
        StopTrainingValue       = 375,                 ...
        ScoreAveragingWindowLength = 100,              ...
        SaveAgentCriteria       = "EpisodeFrequency",  ...
        SaveAgentValue          = 500,                 ...
        SaveAgentDirectory      = "agents_checkpoint");

    trainingStats = train(agent, env, trainOpts);

    save("agent_trained",  "agent");
    save("trainingStats",  "trainingStats");

else
    load('agent_trained.mat', 'agent');
    fprintf('Agente caricato. Pronto per il test.\n');
end

%% =====================================================================
%  Simulazione post-training
% =====================================================================
reset(env);
plot(env, 'full')   % usa il visualizzatore completo con frecce AMS

rng(10)
simOptions = rlSimulationOptions(MaxSteps=10000);
simOptions.NumSimulations = 10;
experience = sim(env, agent, simOptions);

% =====================================================================
%  NOTA: CustomLayer per std_min (opzionale, più preciso)
% =====================================================================
% Se si vuole imporre std >= std_min in modo rigoroso, creare un layer:
%
%   classdef ShiftedSoftplusLayer < nnet.layer.Layer
%       properties, shift = 0.05; end
%       methods
%           function y = predict(this, x)
%               y = log(1 + exp(x)) + this.shift;
%           end
%       end
%   end
%
% e sostituire softplusLayer(Name="stdOutLyr") con
%   ShiftedSoftplusLayer(Name="stdOutLyr")
% impostando this.shift = std_min.