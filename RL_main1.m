clear all
close all
clc

%% Carica ambiente
env = RL_environment1();

doTraining = true;

if doTraining == true

    actInfo = getActionInfo(env);
    obsInfo = getObservationInfo(env);
    numObs  = prod(obsInfo.Dimension);

    criticLayerSizes = [512 256 128];
    actorLayerSizes  = [512 256 128];

    % =====================================================================
    % Critic (Value Function)
    % =====================================================================
    criticNetwork = [
        featureInputLayer(numObs)
        fullyConnectedLayer(criticLayerSizes(1), ...
            Weights=sqrt(2/numObs)*(rand(criticLayerSizes(1),numObs)-0.5), ...
            Bias=1e-3*ones(criticLayerSizes(1),1))
        reluLayer
        fullyConnectedLayer(criticLayerSizes(2), ...
            Weights=sqrt(2/criticLayerSizes(1))*(rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ...
            Bias=1e-3*ones(criticLayerSizes(2),1))
        reluLayer
        fullyConnectedLayer(criticLayerSizes(3), ...
            Weights=sqrt(2/criticLayerSizes(2))*(rand(criticLayerSizes(3),criticLayerSizes(2))-0.5), ...
            Bias=1e-3*ones(criticLayerSizes(3),1))
        reluLayer
        fullyConnectedLayer(1, ...
            Weights=sqrt(2/criticLayerSizes(3))*(rand(1,criticLayerSizes(3))-0.5), ...
            Bias=1e-3)
        ];

    criticNetwork = dlnetwork(criticNetwork);
    summary(criticNetwork)
    critic = rlValueFunction(criticNetwork, obsInfo);

    % =====================================================================
    % Actor (Gaussian Policy)
    % =====================================================================
    inPath = [
        featureInputLayer(numObs, Name="netOin")
        fullyConnectedLayer(actorLayerSizes(1))
        reluLayer
        fullyConnectedLayer(actorLayerSizes(2))
        reluLayer(Name="relulast")
        ];

    meanPath = [
        fullyConnectedLayer(actorLayerSizes(3), Name="MeanLyr")
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension), Name="meanOutLyr")
        tanhLayer(Name="thmeanOutLyr")
        ];

    sdevPath = [
        fullyConnectedLayer(actorLayerSizes(3), Name="StdLyr")
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension))
        reluLayer
        softplusLayer(Name="stdOutLyr")
        ];

    net = layerGraph(inPath);
    net = addLayers(net, meanPath);
    net = addLayers(net, sdevPath);
    net = connectLayers(net, "relulast", "MeanLyr/in");
    net = connectLayers(net, "relulast", "StdLyr/in");
    net = dlnetwork(net);
    summary(net)

    actor = rlContinuousGaussianActor(net, obsInfo, actInfo, ...
        ActionMeanOutputNames="thmeanOutLyr", ...
        ActionStandardDeviationOutputNames="stdOutLyr", ...
        ObservationInputNames="netOin");

    % =====================================================================
    % Opzioni PPO
    %
    % Modifiche rispetto alla versione originale:
    %
    %   ClipFactor:        0.02  → 0.15
    %     Il valore originale era troppo conservativo: aggiornamenti
    %     minuscoli impedivano all'agente di esplorare abbastanza.
    %     0.15 è vicino al default PPO (0.2) e bilancia stabilità/apprendimento.
    %
    %   ExperienceHorizon: 500   → 1024
    %     Con MaxStepsPerEpisode=1000, horizon=500 produceva batch che
    %     coprivano solo metà episodio. A 1024 ogni batch include
    %     quasi sempre almeno un'uscita pacco → gradiente più informativo.
    %
    %   EntropyLossWeight: 0.01  → 0.02
    %     Aumentata leggermente per incentivare l'esplorazione nella fase
    %     iniziale, dove la reward è ancora sparsa.
    %
    %   NumEpoch:          3     → 5
    %     Più passate sullo stesso batch migliorano l'utilizzo dei dati
    %     senza raccogliere nuove esperienze (campionamento efficiente).
    % =====================================================================
    actorOpts  = rlOptimizerOptions(LearnRate=1e-4);
    criticOpts = rlOptimizerOptions(LearnRate=5e-4); % critic converge più veloce dell'actor

    agentOpts = rlPPOAgentOptions(...
        ExperienceHorizon=1024, ...       % era 500
        ClipFactor=0.08, ...              % era 0.02
        EntropyLossWeight=0.02, ...       % era 0.01
        ActorOptimizerOptions=actorOpts, ...
        CriticOptimizerOptions=criticOpts, ...
        NumEpoch=5, ...                   % era 3
        AdvantageEstimateMethod="gae", ...
        GAEFactor=0.95, ...
        SampleTime=0.01, ...
        DiscountFactor=0.99);

    agent = rlPPOAgent(actor, critic, agentOpts);

    % =====================================================================
    % Opzioni di training
    % =====================================================================
    trainOpts = rlTrainingOptions(...
        MaxEpisodes=25000, ...
        MaxStepsPerEpisode=1000, ...
        Plots="training-progress", ...
        StopTrainingCriteria="AverageReward", ...
        StopTrainingValue=40000, ...
        ScoreAveragingWindowLength=100, ...
        SaveAgentCriteria="EpisodeReward", ...
        SaveAgentValue=30000, ...          % salva checkpoint quando episodio > 30k
        SaveAgentDirectory="agents_checkpoint");

    trainingStats = train(agent, env, trainOpts);

    save("agent_trained",   "agent");
    save("trainingStats",   "trainingStats");

else
    load('agent_trained.mat', 'agent');
    fprintf('Agente caricato correttamente. Pronto per il test.\n');
end

%% Simulazione post-training
env.reset();
plot(env)

rng(10)
simOptions = rlSimulationOptions(MaxSteps=10000);
simOptions.NumSimulations = 10;
experience = sim(env, agent, simOptions);