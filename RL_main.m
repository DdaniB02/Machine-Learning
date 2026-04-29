%% RL_MAIN.m  –  PPO Training & Testing for Parcel Singulation
%
%  PROJECT: AMS-based parcel singulation via Proximal Policy Optimisation.
%
%  WORKFLOW
%  --------
%  Set  RUN_MODE = 'train'   to build networks and train from scratch.
%  Set  RUN_MODE = 'resume'  to continue training from a saved checkpoint.
%  Set  RUN_MODE = 'test'    to load a trained agent and run simulations.
%
%  TRAINING ANALYSIS (Section 2)
%  Three PPO configurations are defined to study the effect of key
%  hyperparameters on training performance:
%    Config A  – baseline (recommended starting point)
%    Config B  – larger experience horizon / more update epochs
%    Config C  – smaller clip factor / higher entropy weight (more exploration)
%       costringendo l'agente a provare strategie più diverse e impedendogli di 
%       fossilizzarsi troppo presto su una soluzione sub-ottimale
%
%  EVALUATION METRICS (Section 3)
%  After simulation, the script computes per-episode statistics:
%    - mean reward
%    - number of exits
%    - average inter-parcel spacing at exit
%    - collision rate

clear; close all; clc;

%% ======================================================================
%  0.  CONFIGURATION
%% ======================================================================

RUN_MODE = 'train';          % 'train' | 'resume' | 'test'
PPO_CONFIG = 'A';            % 'A' | 'B' | 'C'  (only used when training)
AGENT_FILE = 'agent_trained.mat';   % file to save / load agent

rng(42);    % reproducibility seed

%% ======================================================================
%  1.  ENVIRONMENT
%% ======================================================================
fprintf('=== Loading RL environment ===\n');
env = RL_environment();

% Quick sanity check
validateEnvironment(env);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
numObs  = prod(obsInfo.Dimension);   % 100
numAct  = prod(actInfo.Dimension);   % 50

fprintf('Observation dim : %d\n', numObs);
fprintf('Action dim      : %d\n', numAct);

%% ======================================================================
%  2.  BUILD AGENT  (train / resume paths)
%% ======================================================================

if strcmp(RUN_MODE, 'test')
    % ---- Load pre-trained agent ----------------------------------------
    fprintf('\n=== Loading trained agent from %s ===\n', AGENT_FILE);
    tmp = load(AGENT_FILE, 'agent');
    agent = tmp.agent;

else
    % ---- Build networks -------------------------------------------------
    fprintf('\n=== Building PPO actor-critic networks ===\n');

    % ---- Critic  (state-value function V(s)) ----------------------------
    criticLayerSizes = [256 256 128];

    criticLayers = [
        featureInputLayer(numObs, Name='obs_in')
        fullyConnectedLayer(criticLayerSizes(1), ...
            Weights = orthogonalInit(criticLayerSizes(1), numObs), ...
            Bias    = zeros(criticLayerSizes(1),1))
        layerNormalizationLayer
        reluLayer
        fullyConnectedLayer(criticLayerSizes(2), ...
            Weights = orthogonalInit(criticLayerSizes(2), criticLayerSizes(1)), ...
            Bias    = zeros(criticLayerSizes(2),1))
        layerNormalizationLayer
        reluLayer
        fullyConnectedLayer(criticLayerSizes(3), ...
            Weights = orthogonalInit(criticLayerSizes(3), criticLayerSizes(2)), ...
            Bias    = zeros(criticLayerSizes(3),1))
        reluLayer
        fullyConnectedLayer(1, ...
            Weights = orthogonalInit(1, criticLayerSizes(3), 1.0), ...
            Bias    = 0)
        ];

    criticNet = dlnetwork(criticLayers);
    critic    = rlValueFunction(criticNet, obsInfo);
    summary(criticNet);

    % ---- Actor  (Gaussian policy) ----------------------------------------
    actorLayerSizes = [256 256 128];

    inPath = [
        featureInputLayer(numObs, Name='obs_in')
        fullyConnectedLayer(actorLayerSizes(1))
        layerNormalizationLayer
        reluLayer
        fullyConnectedLayer(actorLayerSizes(2))
        layerNormalizationLayer
        reluLayer(Name='shared_out')
        ];

    meanPath = [
        fullyConnectedLayer(actorLayerSizes(3), Name='mean_hid')
        reluLayer
        fullyConnectedLayer(numAct, ...
            Weights = orthogonalInit(numAct, actorLayerSizes(3), 0.01), ...
            Bias    = zeros(numAct,1), ...
            Name    = 'mean_lin')
        tanhLayer(Name='mean_out')          % output in (-1,1)
        ];

    stdPath = [
        fullyConnectedLayer(actorLayerSizes(3), Name='std_hid')
        reluLayer
        fullyConnectedLayer(numAct, Name='std_lin')
        reluLayer
        softplusLayer(Name='std_out')       % output > 0
        ];

    actorGraph = layerGraph(inPath);
    actorGraph = addLayers(actorGraph, meanPath);
    actorGraph = addLayers(actorGraph, stdPath);
    actorGraph = connectLayers(actorGraph, 'shared_out', 'mean_hid/in');
    actorGraph = connectLayers(actorGraph, 'shared_out', 'std_hid/in');

    actorNet = dlnetwork(actorGraph);
    actor    = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
                   ActionMeanOutputNames             = 'mean_out', ...
                   ActionStandardDeviationOutputNames = 'std_out', ...
                   ObservationInputNames             = 'obs_in');
    summary(actorNet);

    % ---- PPO agent options  (3 configurations for analysis) -------------
    actorOpts  = rlOptimizerOptions(LearnRate=3e-4, GradientThreshold=0.5);
    criticOpts = rlOptimizerOptions(LearnRate=1e-3, GradientThreshold=0.5);

    switch PPO_CONFIG
        case 'A'
            % Baseline: moderate exploration, standard clipping
            fprintf('\nPPO Config A – Baseline\n');
            agentOpts = rlPPOAgentOptions( ...
                ExperienceHorizon    = 1024, ...
                ClipFactor           = 0.2, ...
                EntropyLossWeight    = 0.01, ...
                ActorOptimizerOptions  = actorOpts, ...
                CriticOptimizerOptions = criticOpts, ...
                NumEpoch             = 4, ...
                AdvantageEstimateMethod = 'gae', ...
                GAEFactor            = 0.95, ...
                SampleTime           = env.dt, ...
                DiscountFactor       = 0.99, ...
                MiniBatchSize        = 256);

        case 'B'
            % Longer horizon + more update epochs -> better sample efficiency
            fprintf('\nPPO Config B – Long horizon / more epochs\n');
            agentOpts = rlPPOAgentOptions( ...
                ExperienceHorizon    = 2048, ...
                ClipFactor           = 0.2, ...
                EntropyLossWeight    = 0.005, ...
                ActorOptimizerOptions  = rlOptimizerOptions(LearnRate=1e-4, GradientThreshold=0.5), ...
                CriticOptimizerOptions = rlOptimizerOptions(LearnRate=5e-4, GradientThreshold=0.5), ...
                NumEpoch             = 8, ...
                AdvantageEstimateMethod = 'gae', ...
                GAEFactor            = 0.97, ...
                SampleTime           = env.dt, ...
                DiscountFactor       = 0.995, ...
                MiniBatchSize        = 512);

        case 'C'
            % High entropy + tight clip -> more exploration, slower convergence
            fprintf('\nPPO Config C – High exploration\n');
            agentOpts = rlPPOAgentOptions( ...
                ExperienceHorizon    = 512, ...
                ClipFactor           = 0.1, ...
                EntropyLossWeight    = 0.05, ...
                ActorOptimizerOptions  = rlOptimizerOptions(LearnRate=5e-4, GradientThreshold=0.5), ...
                CriticOptimizerOptions = rlOptimizerOptions(LearnRate=1e-3, GradientThreshold=0.5), ...
                NumEpoch             = 3, ...
                AdvantageEstimateMethod = 'gae', ...
                GAEFactor            = 0.90, ...
                SampleTime           = env.dt, ...
                DiscountFactor       = 0.99, ...
                MiniBatchSize        = 128);

        otherwise
            error('Unknown PPO_CONFIG. Choose A, B or C.');
    end

    agent = rlPPOAgent(actor, critic, agentOpts);

    % ---- Resume: load existing weights ----------------------------------
    if strcmp(RUN_MODE, 'resume') && isfile(AGENT_FILE)
        fprintf('Resuming from %s ...\n', AGENT_FILE);
        tmp   = load(AGENT_FILE, 'agent');
        agent = tmp.agent;
    end

    % ---- Training options -----------------------------------------------
    
    trainOpts = rlTrainingOptions( ...
        MaxEpisodes              = 20000, ...
        MaxStepsPerEpisode       = 1500, ...
        Plots                    = 'training-progress', ...
        Verbose                  = true, ... %%VerboseFrequency         = 50, ...
        StopTrainingCriteria     = 'AverageReward', ...
        StopTrainingValue        = 30000, ...
        ScoreAveragingWindowLength = 100, ...
        SaveAgentCriteria        = 'EpisodeReward', ...
        SaveAgentValue           = 10000, ...
        SaveAgentDirectory       = 'saved_agents');

    fprintf('\n=== Starting PPO training (Config %s) ===\n', PPO_CONFIG);
    trainingStats = train(agent, env, trainOpts);

    % Save
    save(AGENT_FILE, 'agent');
    save('trainingStats.mat', 'trainingStats');
    fprintf('\nAgent saved to %s\n', AGENT_FILE);

    % Plot training curve
    plotTrainingCurve(trainingStats, PPO_CONFIG);
end

%% ======================================================================
%  3.  SIMULATION & EVALUATION
%% ======================================================================
fprintf('\n=== Running post-training simulations ===\n');

N_SIM   = 20;       % number of evaluation episodes
MAX_STP = 2000;     % max steps per episode

env.reset();
plot(env);          % open visualiser

rng(123);
simOptions = rlSimulationOptions('MaxSteps', MAX_STP, 'NumSimulations', N_SIM);
experience = sim(env, agent, simOptions);

% ---- Compute evaluation metrics ----------------------------------------
metrics = evaluateSingulation(experience, N_SIM, env.max_gen_boxes);

fprintf('\n======= SINGULATION PERFORMANCE (%d episodes) =======\n', N_SIM);
fprintf('  Mean episode reward      : %8.1f  (std %.1f)\n', ...
        metrics.mean_reward,   metrics.std_reward);
fprintf('  Mean exits per episode   : %8.2f  (max %d)\n', ...
        metrics.mean_exits,    env.max_gen_boxes);
fprintf('  Singulation rate         : %8.1f %%\n', ...
        metrics.singulation_rate*100);
fprintf('  Mean steps to completion : %8.1f\n', ...
        metrics.mean_steps);
fprintf('=====================================================\n');

%% ======================================================================
%  LOCAL FUNCTIONS
%% ======================================================================

function W = orthogonalInit(rows, cols, scale)
%ORTHOGONALINIT  Orthogonal weight initialisation (better than random for PPO).
    if nargin < 3, scale = sqrt(2); end
    if rows < cols
        W = orth(randn(cols, rows))';
    else
        W = orth(randn(rows, cols));
    end
    W = scale * W(1:rows, 1:cols);
end

function validateEnvironment(env)
%VALIDATEENVIRONMENT  Quick smoke-test of environment step/reset.
    fprintf('Validating environment ... ');
    obs0 = reset(env);
    assert(numel(obs0) == 100, 'Observation size mismatch.');
    act0 = zeros(50,1);
    [obs1, rew1, done1, ~] = step(env, act0);
    assert(numel(obs1) == 100, 'Step observation size mismatch.');
    assert(isscalar(rew1),     'Reward must be scalar.');
    assert(islogical(done1) || isnumeric(done1), 'IsDone type error.');
    fprintf('OK\n');
end

function plotTrainingCurve(stats, cfg_label)
%PLOTTRAININGCURVE  Plot episode reward and smoothed average.
    if ~isstruct(stats), return; end
    figure('Name', sprintf('Training – Config %s', cfg_label), ...
           'NumberTitle', 'off');

    ep  = stats.EpisodeIndex;
    rew = stats.EpisodeReward;
    avg = stats.AverageReward;

    yyaxis left
    plot(ep, rew, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
    hold on;
    plot(ep, avg, 'b', 'LineWidth', 2);
    ylabel('Reward');

    yyaxis right
    if isfield(stats, 'EpisodeSteps')
        plot(ep, stats.EpisodeSteps, 'r--', 'LineWidth', 1);
        ylabel('Steps per episode');
    end

    xlabel('Episode');
    title(sprintf('PPO Training – Config %s', cfg_label));
    legend({'Episode reward', 'Average reward (100-ep)', 'Steps'}, ...
           'Location', 'northwest');
    grid on;
    hold off;
end

function metrics = evaluateSingulation(experience, N_sim, max_gen)
%EVALUATESINGULATION  Extract per-episode metrics from sim() output.
%
%  Returns a struct with fields:
%    mean_reward, std_reward, mean_exits, singulation_rate, mean_steps

    rewards  = zeros(N_sim,1);
    exits    = zeros(N_sim,1);
    n_steps  = zeros(N_sim,1);

    for k = 1:min(N_sim, numel(experience))
        ep = experience(k);
        r  = ep.Reward.Data;
        rewards(k) = sum(r);
        n_steps(k) = numel(r);

        % Count exits from observation: feature 9 (exited flag) of each parcel
        % obs layout: 10 features per parcel, feature 9 = exited
        obs_final = ep.Observation{1}.Data(:,:,end);  % last timestep
        n_ex = 0;
        for p = 1:max_gen
            exited_flag = obs_final(10*(p-1)+9);
            if exited_flag > 0.5
                n_ex = n_ex + 1;
            end
        end
        exits(k) = n_ex;
    end

    metrics.mean_reward       = mean(rewards);
    metrics.std_reward        = std(rewards);
    metrics.mean_exits        = mean(exits);
    metrics.singulation_rate  = mean(exits == max_gen);
    metrics.mean_steps        = mean(n_steps);
end
