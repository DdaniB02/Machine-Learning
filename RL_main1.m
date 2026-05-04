
clear all
close all
clc

%%

% load the RL environment

env=RL_environment1();

doTraining = false;

if doTraining == true

    actInfo = getActionInfo(env);
    obsInfo = getObservationInfo(env);

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
        ];

    criticNetwork = dlnetwork(criticNetwork);
    summary(criticNetwork)

    critic = rlValueFunction(criticNetwork,obsInfo);

    % Actor

    inPath = [
        featureInputLayer(numObs,Name="netOin")
        fullyConnectedLayer(actorLayerSizes(1))
        reluLayer
        fullyConnectedLayer(actorLayerSizes(2))
        reluLayer(Name="relulast")
        ];

    meanPath = [
        fullyConnectedLayer(actorLayerSizes(3),Name="MeanLyr")
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension),Name="meanOutLyr")
        tanhLayer(Name="thmeanOutLyr");
        ];

    sdevPath = [
        fullyConnectedLayer(actorLayerSizes(3),Name="StdLyr")
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension))
        reluLayer
        softplusLayer(Name="stdOutLyr")
        ];

    % Add layers to network object
    net = layerGraph(inPath);
    net = addLayers(net,meanPath);
    net = addLayers(net,sdevPath);

    % Connect layers
    net = connectLayers(net,"relulast","MeanLyr/in");
    net = connectLayers(net,"relulast","StdLyr/in");

    net = dlnetwork(net);
    summary(net)

    actor = rlContinuousGaussianActor(net, obsInfo, actInfo, ...
        ActionMeanOutputNames="thmeanOutLyr",...
        ActionStandardDeviationOutputNames="stdOutLyr",...
        ObservationInputNames="netOin");

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

    agent = rlPPOAgent(actor,critic,agentOpts);

    trainOpts = rlTrainingOptions(...
        MaxEpisodes=25000,...
        MaxStepsPerEpisode=1000,...
        Plots="training-progress",...
        StopTrainingCriteria="AverageReward",...
        StopTrainingValue=40000,...
        ScoreAveragingWindowLength=100);

    trainingStats = train(agent, env, trainOpts);

    save("agent_trained","agent");
    save("trainingStats","trainingStats");

else

    % load saved agent
    % Carica l'agente precedentemente salvato
    load('agent_trained.mat', 'agent');
    
    % (Opzionale) Visualizza l'agente per conferma
    fprintf('Agente caricato correttamente. Pronto per il test.\n');

end

%% simulation of the sorting after training

env.reset();

plot(env)

rng(10)
simOptions = rlSimulationOptions(MaxSteps=10000);
simOptions.NumSimulations = 10;
experience = sim(env, agent, simOptions);
