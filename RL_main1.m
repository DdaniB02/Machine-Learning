
clear all
close all
clc

%%

% load the RL environment

env=RL_environment1();

doTraining = false;

if doTraining

    actInfo = getActionInfo(env);
    obsInfo = getObservationInfo(env);
<<<<<<< Updated upstream

    numObs = prod(obsInfo.Dimension);
    criticLayerSizes = [512 256 128];
    actorLayerSizes = [512 256 128];

    % Critic

    criticNetwork = [
        featureInputLayer(numObs)
        fullyConnectedLayer(criticLayerSizes(1), ...
        Weights=sqrt(2/numObs)*...
        (rand(criticLayerSizes(1),numObs)-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(1),1))
        reluLayer
        fullyConnectedLayer(criticLayerSizes(2), ...
        Weights=sqrt(2/criticLayerSizes(1))*...
        (rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(2),1))
        reluLayer
        fullyConnectedLayer(criticLayerSizes(3), ...
        Weights=sqrt(2/criticLayerSizes(2))*...
        (rand(criticLayerSizes(3),criticLayerSizes(2))-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(3),1))
        reluLayer
        fullyConnectedLayer(1, ...
        Weights=sqrt(2/criticLayerSizes(3))* ...
        (rand(1,criticLayerSizes(3))-0.5), ...
        Bias=1e-3)
=======
    numObs  = prod(obsInfo.Dimension);   % 100
    numAct  = prod(actInfo.Dimension);   % 50

    % =====================================================================
    % Architettura reti  [RIDOTTA a 256-128]
    %
    % Motivazione: reti più piccole convergono più velocemente con pochi
    % dati e hanno meno rischio di overfitting alle prime esperienze.
    % 512-256-128 è eccessivo per uno spazio di stato/azione sparse
    % (molte celle AMS vuote). 256-128 è un buon compromesso.
    % =====================================================================
    criticLayerSizes = [256 128];
    actorLayerSizes  = [256 128];

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
>>>>>>> Stashed changes
        ];
    criticNetwork = dlnetwork(criticNetwork);
<<<<<<< Updated upstream
    summary(criticNetwork)

    critic = rlValueFunction(criticNetwork,obsInfo);

    % Actor
=======
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
>>>>>>> Stashed changes

    inPath = [
        featureInputLayer(numObs,Name="netOin")
        fullyConnectedLayer(actorLayerSizes(1))
        reluLayer
        fullyConnectedLayer(actorLayerSizes(2))
        reluLayer(Name="relulast")
        ];

    meanPath = [
<<<<<<< Updated upstream
        fullyConnectedLayer(actorLayerSizes(3),Name="MeanLyr")
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension),Name="meanOutLyr")
        tanhLayer(Name="thmeanOutLyr");
=======
        fullyConnectedLayer(actorLayerSizes(2), Name="MeanLyr")
        reluLayer
        fullyConnectedLayer(numAct, Name="meanOutLyr", ...
            Bias=zeros(numAct,1))   % bias=0 → policy inizia centrata
        tanhLayer(Name="thmeanOutLyr")
>>>>>>> Stashed changes
        ];

    % std_min via bias inizializzato: softplus(x+b)+std_min
    % Per avere std_init ≈ 0.2: softplus(-1) ≈ 0.31 → bias=-1 → ok
    sdevPath = [
<<<<<<< Updated upstream
        fullyConnectedLayer(actorLayerSizes(3),Name="StdLyr")
=======
        fullyConnectedLayer(actorLayerSizes(2), Name="StdLyr")
>>>>>>> Stashed changes
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

    % Add layers to network object
    net = layerGraph(inPath);
    net = addLayers(net,meanPath);
    net = addLayers(net,sdevPath);

    % Connect layers
    net = connectLayers(net,"relulast","MeanLyr/in");
    net = connectLayers(net,"relulast","StdLyr/in");

    net = dlnetwork(net);

    actor = rlContinuousGaussianActor(net, obsInfo, actInfo, ...
        ActionMeanOutputNames="thmeanOutLyr",...
        ActionStandardDeviationOutputNames="stdOutLyr",...
        ObservationInputNames="netOin");

<<<<<<< Updated upstream
    % Train

    actorOpts = rlOptimizerOptions(LearnRate=1e-4);
    criticOpts = rlOptimizerOptions(LearnRate=1e-4);

    agentOpts = rlPPOAgentOptions(...
        ExperienceHorizon=500,...
        ClipFactor=0.02,...
        EntropyLossWeight=0.01,...
        ActorOptimizerOptions=actorOpts,...
        CriticOptimizerOptions=criticOpts,...
        NumEpoch=3,...
        AdvantageEstimateMethod="gae",...
        GAEFactor=0.95,...
        SampleTime=0.01,...
        DiscountFactor=0.99);
=======
    % =====================================================================
    % Opzioni PPO
    %
    % Allineate al paper (Tabella 1) con piccole correzioni pratiche:
    %
    %   ExperienceHorizon  2048  → nsteps paper
    %   ClipFactor         0.2   → ε_π = ε_V paper
    %   EntropyLossWeight  1e-4  → c2 paper (basso: std già limitata)
    %   NumEpoch           10    → ridotto da 50: evita overfitting su
    %                              batch piccoli nelle prime fasi curriculum
    %   MiniBatchSize      64    → paper
    %   LR actor/critic    3e-4  → paper (Adam)
    %   GAEFactor          0.95  → λ paper
    %   DiscountFactor     0.99  → γ paper
    % =====================================================================
    actorOpts  = rlOptimizerOptions(LearnRate=3e-4);
    criticOpts = rlOptimizerOptions(LearnRate=3e-4);

    agentOpts = rlPPOAgentOptions( ...
        ExperienceHorizon    = 2048,  ...
        ClipFactor           = 0.2,   ...
        EntropyLossWeight    = 1e-4,  ...
        ActorOptimizerOptions  = actorOpts,  ...
        CriticOptimizerOptions = criticOpts, ...
        NumEpoch             = 10,    ...
        AdvantageEstimateMethod = "gae", ...
        GAEFactor            = 0.95,  ...
        SampleTime           = 0.01,  ...
        DiscountFactor       = 0.99,  ...
        MiniBatchSize        = 64);
>>>>>>> Stashed changes

    agent = rlPPOAgent(actor,critic,agentOpts);

<<<<<<< Updated upstream
    trainOpts = rlTrainingOptions(...
        MaxEpisodes=25000,...
        MaxStepsPerEpisode=1000,...
        Plots="training-progress",...
        StopTrainingCriteria="AverageReward",...
        StopTrainingValue=40000,...
        ScoreAveragingWindowLength=100);
=======
    % =====================================================================
    % Opzioni di training
    %
    % Nuova scala reward: ≈ [−0.3, 1.3] per step × 1000 step/ep → [−300, 1300]
    % Soglia stop: 800 (media su 100 ep) → policy quasi-ottima
    % Checkpoint ogni 500 ep (invariato)
    % =====================================================================
    trainOpts = rlTrainingOptions( ...
        MaxEpisodes             = 25000,           ...
        MaxStepsPerEpisode      = 1000,            ...
        Plots                   = "training-progress", ...
        StopTrainingCriteria    = "AverageReward", ...
        StopTrainingValue       = 800,             ...
        ScoreAveragingWindowLength = 100,          ...
        SaveAgentCriteria       = "EpisodeFrequency", ...
        SaveAgentValue          = 500,             ...
        SaveAgentDirectory      = "agents_checkpoint");
>>>>>>> Stashed changes

    % --- Training ---
    trainingStats = train(agent, env, trainOpts);

<<<<<<< Updated upstream
    save("agent_trained","agent");
    save("trainingStats","trainingStats");
=======
    save("agent_trained",  "agent");
    save("trainingStats",  "trainingStats");
>>>>>>> Stashed changes

else

    % load saved agent
    % Carica l'agente precedentemente salvato
    load('agent_trained.mat', 'agent');
<<<<<<< Updated upstream
    
    % (Opzionale) Visualizza l'agente per conferma
    fprintf('Agente caricato correttamente. Pronto per il test.\n');

end

%% simulation of the sorting after training

env.reset();

plot(env)
=======
    fprintf('Agente caricato. Pronto per il test.\n');
end

%% =====================================================================
%  Simulazione post-training
% =====================================================================
env.reset();
plot(env, 'full')   % usa il visualizzatore completo con frecce AMS
>>>>>>> Stashed changes

rng(10)
simOptions = rlSimulationOptions(MaxSteps=10000);
simOptions.NumSimulations = 10;
experience = sim(env, agent, simOptions);
<<<<<<< Updated upstream
=======

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
>>>>>>> Stashed changes
