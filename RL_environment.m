classdef RL_environment < rl.env.MATLABEnvironment
%RL_ENVIRONMENT  Custom MATLAB RL environment for parcel singulation
%  on a 5x5 Autonomous Mobile Sorter (AMS) matrix.
%
% -----------------------------------------------------------------------
%  PHYSICAL SETUP
% -----------------------------------------------------------------------
%  A 5x5 grid of AMS cells (each d_AMS x d_AMS = 0.2 x 0.2 m) forms a
%  1 m x 1 m sorting surface. Parcels enter at the bottom (y~0) and must
%  exit singulated (one per lane) at the top (y > 1 m).
%
% -----------------------------------------------------------------------
%  STATE  (100 elements, all normalised to [0,1])
% -----------------------------------------------------------------------
%  10 parcel slots x 10 features each:
%    1  active    : 1 if parcel is currently on the AMS surface
%    2  x_norm    : x / l_AMS
%    3  y_norm    : y / l_AMS
%    4  d_norm    : parcel size / (2*d_AMS)
%    5  vx_norm   : (vx + v_max)/(2*v_max)  -- centred, normalised
%    6  vy_norm   : (vy + v_max)/(2*v_max)
%    7  j_norm    : AMS column / n_j_AMS  (0 if not on AMS)
%    8  i_norm    : AMS row    / n_i_AMS
%    9  exited    : 1 once the parcel has left the AMS
%   10  spacing   : normalised min edge-to-edge distance to nearest parcel
%
% -----------------------------------------------------------------------
%  ACTION  (50 elements, PPO outputs in [-1,1] -> de-normalised inside step)
% -----------------------------------------------------------------------
%  25 AMS cells x 2 values:
%    odd  indices -> rotation  in [-45 deg, +45 deg]
%    even indices -> velocity  in [0.5,  2.2] m/s
%
% -----------------------------------------------------------------------
%  REWARD  (multi-component, see getReward)
% -----------------------------------------------------------------------
%  +w_throughput  per new parcel that exits
%  +w_spacing     proportional to normalised minimum inter-parcel gap
%  +w_progress    proportional to total forward (y) displacement per step
%  +w_collision   (negative) per pairwise collision resolved in this step
%  +w_idle        (negative) when no parcel is active on the AMS

    %% ================================================================
    %  Properties
    %% ================================================================
    properties
        % Physical parameters
        v_treadmill   = 0.6;            % m/s  -- belt speed beyond AMS
        d_AMS         = 0.2;            % m    -- AMS cell size
        n_i_AMS       = 5;              % rows  (y-direction)
        n_j_AMS       = 5;              % cols  (x-direction)
        n_actions_art = 2;              % actions per AMS cell

        upperlim_rot  =  pi*45/180;     % rad
        upperlim_v    =  2.2;           % m/s
        lowerlim_rot  = -pi*45/180;     % rad
        lowerlim_v    =  0.5;           % m/s

        ActLimUpVec  = zeros(50,1);
        ActLimLowVec = zeros(50,1);

        % Parcel kinematics
        vy_boxes      = [];
        x_boxes_prec  = [];
        y_boxes_prec  = [];
        index_AMS     = [];
        LastAction    = zeros(50,1);
        i_box         = [];
        j_box         = [];

        % Parcel bookkeeping
        n_boxes_tot_vect = [];
        n_boxes_tot   = 0;
        max_gen_boxes = 10;
        cont_new_box  = 0;

        d_boxes       = zeros(10,1);
        x_boxes       = zeros(10,1);
        y_boxes       = zeros(10,1);
        x_boxes_vect  = zeros(10000,10);
        y_boxes_vect  = zeros(10000,10);
        cont          = 0;

        pack_exited   = zeros(10,1);
        exit_order    = zeros(10,1);
        index_exit    = 0;

        toll_contatto = 0.005;          % m
        dt            = 0.01;           % s
        time          = 0;              % s

        % Reward weights
        w_throughput  = 500;    % bonus per exiting parcel       [tune: 200-1000]
        w_spacing     = 15;     % spacing quality scale          [tune:  5-30  ]
        w_progress    = 3;      % forward progress per step      [tune:  1-10  ]
        w_collision   = -30;    % penalty per collision          [tune: -10 to -100]
        w_idle        = -1;     % penalty when AMS empty         [tune: -0.1 to -5]

        % Internal reward helpers
        prev_y_boxes     = zeros(10,1);
        prev_pack_exited = zeros(10,1);
        n_collisions     = 0;
    end

    properties (Hidden)
        VisualizeAnimation = true
        VisualizeActions   = false
        VisualizeStates    = false
    end

    properties
        State = zeros(100,1)
    end

    properties (Access = protected)
        IsDone = false
    end

    properties (Transient, Access = private)
        Visualizer = []
    end

    %% ================================================================
    %  Constructor
    %% ================================================================
    methods
        function this = RL_environment()

            n_i_AMS_local = 5;
            n_j_AMS_local = 5;
            n_actions = n_i_AMS_local * n_j_AMS_local * 2;

            ObservationInfo = rlNumericSpec([100 1]);
            ObservationInfo.Name = 'AMS System State';
            ObservationInfo.LowerLimit = -ones(100,1);
            ObservationInfo.UpperLimit =  ones(100,1);

            ActionInfo = rlNumericSpec([n_actions 1]);
            ActionInfo.Name = 'AMS Actions';
            ActionInfo.LowerLimit = -ones(n_actions,1);
            ActionInfo.UpperLimit =  ones(n_actions,1);

            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);

            lim_up  = zeros(n_actions,1);
            lim_low = zeros(n_actions,1);

            for ii = 1:2:n_actions
                lim_up(ii)    = this.upperlim_rot;
                lim_up(ii+1)  = this.upperlim_v;
                lim_low(ii)   = this.lowerlim_rot;
                lim_low(ii+1) = this.lowerlim_v;
            end

            this.ActLimUpVec  = lim_up;
            this.ActLimLowVec = lim_low;
        end

        %% =============================================================
        %  step
        %% =============================================================
        function [Observation, Reward, IsDone, Info] = step(this, Action)
            Info   = [];
            this.time = this.time + this.dt;
            this.cont = this.cont + 1;

            l_AMS  = this.d_AMS * this.n_j_AMS;
            y_exit = this.d_AMS * this.n_i_AMS;

            % De-normalise actions (PPO output in [-1,1])
            lo = zeros(50,1);  hi = zeros(50,1);
            for ii = 1:2:50
                lo(ii)   = this.lowerlim_rot;  hi(ii)   = this.upperlim_rot;
                lo(ii+1) = this.lowerlim_v;    hi(ii+1) = this.upperlim_v;
            end
            AMS_actions = lo + (1 + Action) .* (hi - lo) / 2;
            AMS_actions = max(lo, min(hi, AMS_actions));
            this.LastAction = AMS_actions;

            % Build rotation/velocity matrices
            v_AMS   = zeros(5,5);
            rot_AMS = zeros(5,5);
            for ii = 1:this.n_i_AMS
                for jj = 1:this.n_j_AMS
                    rot_AMS(ii,jj) = AMS_actions(jj*2-1 + (ii-1)*10);
                    v_AMS(ii,jj)   = AMS_actions(jj*2   + (ii-1)*10);
                end
            end

            % Spawn new parcels every 0.75 s
            if mod(round(this.time,2), 0.75) == 0 && ...
                    this.n_boxes_tot < this.max_gen_boxes && this.time > 0.25
                this.cont_new_box = this.cont_new_box + 1;
                gen_n = min(2, this.max_gen_boxes - this.n_boxes_tot);
                this.n_boxes_tot = this.n_boxes_tot + gen_n;
                this.n_boxes_tot_vect(this.cont_new_box) = gen_n;

                for ii = 1:gen_n
                    if gen_n > 1
                        if ii == 1
                            idx = this.n_boxes_tot - 1;
                            this.x_boxes(idx) = this.d_boxes(idx)*0.5 + ...
                                (this.d_AMS*2.5 - this.d_boxes(idx)*0.5)*rand();
                            this.y_boxes(idx) = 0.001 + rand()*0.01;
                        else
                            idx  = this.n_boxes_tot;
                            idxp = this.n_boxes_tot - 1;
                            x_new = this.d_AMS*2.5 + this.d_boxes(idx)*0.5 + ...
                                (l_AMS - this.d_AMS*2.5 - this.d_boxes(idx)*0.5)*rand();
                            rsum = this.d_boxes(idx)/2 + this.d_boxes(idxp)/2;
                            if abs(x_new - this.x_boxes(idxp)) <= rsum
                                x_new = this.x_boxes(idxp) + rsum + 0.1;
                            end
                            x_new = max(this.d_boxes(idx)/2, ...
                                    min(l_AMS - this.d_boxes(idx)/2, x_new));
                            if abs(x_new - this.x_boxes(idxp)) <= rsum
                                this.x_boxes(idxp) = x_new - rsum - 0.1;
                            end
                            this.x_boxes(idx) = x_new;
                            this.y_boxes(idx)  = 0.001 + rand()*0.05;
                        end
                    else
                        idx = this.n_boxes_tot;
                        this.x_boxes(idx) = this.d_boxes(idx)*0.55 + ...
                            (l_AMS - this.d_boxes(idx)*0.55)*rand();
                        this.y_boxes(idx) = 0.001 + rand()*0.01;
                    end
                    this.x_boxes_prec(idx) = this.x_boxes(idx);
                    this.y_boxes_prec(idx) = this.y_boxes(idx);
                end
            end

            % Inactive slots: sentinel value
            for ii = this.n_boxes_tot+1 : this.max_gen_boxes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end

            % Snapshot for reward computation
            this.prev_y_boxes     = this.y_boxes;
            this.prev_pack_exited = this.pack_exited;
            this.n_collisions     = 0;

            % Kinematics + collision resolution
            for ii = 1:this.n_boxes_tot

                % Wall clamp (x)
                if this.x_boxes(ii) - this.d_boxes(ii)/2 < 0
                    this.x_boxes(ii) = this.d_boxes(ii)/2;
                elseif this.x_boxes(ii) + this.d_boxes(ii)/2 > l_AMS
                    this.x_boxes(ii) = l_AMS - this.d_boxes(ii)/2;
                end

                % AMS cell
                row = ceil(this.y_boxes(ii) / this.d_AMS);
                this.i_box(ii) = max(1, min(6, row));
                if this.i_box(ii) >= 6
                    this.j_box(ii) = 6;
                else
                    col = ceil(this.x_boxes(ii) / this.d_AMS);
                    this.j_box(ii) = max(1, min(5, col));
                end

                % Move
                if this.i_box(ii) < 6
                    r = rot_AMS(this.i_box(ii), this.j_box(ii));
                    v = v_AMS(this.i_box(ii),   this.j_box(ii));
                    this.x_boxes(ii)  = this.x_boxes(ii) + v*this.dt*sin(r);
                    this.y_boxes(ii)  = this.y_boxes(ii) + v*this.dt*cos(r);
                    this.vy_boxes(ii) = v*cos(r);
                else
                    this.y_boxes(ii)  = this.y_boxes(ii) + this.v_treadmill*this.dt;
                    this.vy_boxes(ii) = this.v_treadmill;
                end

                % Pairwise collision resolution
                if ii > 1 && this.cont > 1
                    ct = this.cont - 1;
                    for jj = ii:-1:2
                        if this.x_boxes(jj-1) < 0, continue; end
                        rsum = this.d_boxes(ii)/2 + this.d_boxes(jj-1)/2;
                        dx_c = abs(this.x_boxes(ii) - this.x_boxes(jj-1));
                        dy_c = abs(this.y_boxes(ii) - this.y_boxes(jj-1));
                        dx_p = abs(this.x_boxes_vect(ii,ct) - this.x_boxes_vect(jj-1,ct));
                        dy_p = abs(this.y_boxes_vect(ii,ct) - this.y_boxes_vect(jj-1,ct));

                        if dx_c <= rsum+0.001 && dx_p >= rsum && dy_c <= rsum && dy_p <= rsum
                            pen = rsum - dx_c + this.toll_contatto;
                            sgn = sign(this.x_boxes(ii) - this.x_boxes(jj-1));
                            if sgn == 0, sgn = 1; end
                            this.x_boxes(ii)   = this.x_boxes(ii)   + sgn*pen/2;
                            this.x_boxes(jj-1) = this.x_boxes(jj-1) - sgn*pen/2;
                            this.n_collisions  = this.n_collisions + 1;
                        elseif dy_c <= rsum+0.001 && dy_p >= rsum && dx_c <= rsum
                            pen = rsum - dy_c + this.toll_contatto;
                            sgn = sign(this.y_boxes(ii) - this.y_boxes(jj-1));
                            if sgn == 0, sgn = 1; end
                            this.y_boxes(ii)   = this.y_boxes(ii)   + sgn*pen/2;
                            this.y_boxes(jj-1) = this.y_boxes(jj-1) - sgn*pen/2;
                            this.n_collisions  = this.n_collisions + 1;
                        end
                    end
                end

                if this.i_box(ii) < 6 && this.j_box(ii) < 6
                    this.index_AMS(ii) = this.j_box(ii) + (this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end
            end

            % Trajectory history
            for ii = 1:this.max_gen_boxes
                if ii <= this.n_boxes_tot
                    this.x_boxes_vect(ii, this.cont) = this.x_boxes(ii);
                    this.y_boxes_vect(ii, this.cont) = this.y_boxes(ii);
                    this.x_boxes_prec(ii) = this.x_boxes(ii);
                    this.y_boxes_prec(ii) = this.y_boxes(ii);
                else
                    this.x_boxes_vect(ii, this.cont) = -1;
                    this.y_boxes_vect(ii, this.cont) = -1;
                end
            end

            % Terminal condition
            cont_exit = 0;
            for ii = 1:this.n_boxes_tot
                if this.y_boxes(ii) > y_exit
                    cont_exit = cont_exit + 1;
                    if this.pack_exited(ii) == 0
                        this.pack_exited(ii) = 1;
                        this.index_exit = this.index_exit + 1;
                        this.exit_order(this.index_exit) = ii;
                    end
                end
            end

            IsDone      = (cont_exit == this.max_gen_boxes);
            this.IsDone = IsDone;

            Observation = buildObservation(this);
            this.State  = Observation;
            Reward      = getReward(this);

            notifyEnvUpdated(this);
        end

        %% =============================================================
        %  reset
        %% =============================================================
        function InitialObservation = reset(this)

            this.index_exit        = 0;
            this.pack_exited       = zeros(10,1);
            this.prev_pack_exited  = zeros(10,1);
            this.exit_order        = zeros(10,1);
            this.cont_new_box      = 1;
            this.n_boxes_tot_vect  = 2;
            this.n_boxes_tot       = 2;
            this.time              = 0;
            this.cont              = 0;
            this.n_collisions      = 0;
            this.vy_boxes          = zeros(this.max_gen_boxes, 1);
            this.prev_y_boxes      = zeros(10,1);
            this.x_boxes_vect      = zeros(10000, 10);
            this.y_boxes_vect      = zeros(10000, 10);

            l_AMS = this.d_AMS * this.n_j_AMS;
            max_d = 2.0 * this.d_AMS;
            min_d = 0.25 * this.d_AMS;

            for ii = 1:this.max_gen_boxes
                this.d_boxes(ii) = min_d + (max_d - min_d)*rand();
            end

            % Place 2 initial parcels without overlap
            x1 = this.d_boxes(1)*0.5 + (this.d_AMS*2.5 - this.d_boxes(1)*0.5)*rand();
            x2 = this.d_AMS*2.5 + this.d_boxes(2)*0.5 + ...
                 (l_AMS - this.d_AMS*2.5 - this.d_boxes(2)*0.5)*rand();
            rsum12 = this.d_boxes(1)/2 + this.d_boxes(2)/2;
            if abs(x2-x1) <= rsum12
                x2 = x1 + rsum12 + 0.1;
            end
            x2 = max(this.d_boxes(2)/2, min(l_AMS - this.d_boxes(2)/2, x2));
            if abs(x2-x1) <= rsum12
                x1 = x2 - rsum12 - 0.1;
            end
            y1 = 0.001 + rand()*0.01;
            y2 = 0.001 + rand()*0.05;

            this.x_boxes(1) = x1;  this.y_boxes(1) = y1;
            this.x_boxes(2) = x2;  this.y_boxes(2) = y2;
            this.x_boxes_prec = zeros(1, this.max_gen_boxes);
            this.y_boxes_prec = zeros(1, this.max_gen_boxes);
            this.x_boxes_prec(1) = x1;  this.x_boxes_prec(2) = x2;
            this.y_boxes_prec(1) = y1;  this.y_boxes_prec(2) = y2;

            for ii = this.n_boxes_tot+1 : this.max_gen_boxes
                this.x_boxes(ii) = 0;
                this.y_boxes(ii) = 0;
            end

            this.index_AMS = zeros(this.max_gen_boxes, 1);
            for ii = 1:this.n_boxes_tot
                row = ceil(this.y_boxes(ii) / this.d_AMS);
                this.i_box(ii) = max(1, min(6, row));
                if this.i_box(ii) >= 6
                    this.j_box(ii) = 6;
                else
                    col = ceil(this.x_boxes(ii) / this.d_AMS);
                    this.j_box(ii) = max(1, min(5, col));
                end
                if this.i_box(ii) < 6 && this.j_box(ii) < 6
                    this.index_AMS(ii) = this.j_box(ii) + (this.i_box(ii)-1)*5;
                end
            end

            % Seed history
            for ii = 1:this.n_boxes_tot
                this.x_boxes_vect(1, ii) = this.x_boxes(ii);
                this.y_boxes_vect(1, ii) = this.y_boxes(ii);
            end

            InitialObservation = buildObservation(this);
            this.State = InitialObservation;
            notifyEnvUpdated(this);
        end
    end

    %% ================================================================
    %  Public methods
    %% ================================================================
    methods

        function varargout = plot(this)
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = AMSVisualizer(this);
            else
                bringToFront(this.Visualizer);
            end
            if nargout
                varargout{1} = this.Visualizer;
            end
            this.VisualizeAnimation = true;
            this.VisualizeActions   = false;
            this.VisualizeStates    = false;
        end

        function Reward = getReward(this)
        %GETREWARD  Multi-component reward for parcel singulation.
        %
        %  COMPONENT 1 – Throughput (+w_throughput)
        %    Large positive bonus each time a NEW parcel exits. This is the
        %    primary learning signal: the agent must push all 10 parcels out.
        %
        %  COMPONENT 2 – Spacing / singulation (+w_spacing x gap_score)
        %    Encourages lateral separation. gap_score = normalised minimum
        %    edge-to-edge gap. With 1 parcel on belt -> full bonus (perfect
        %    singulation). With many overlapping parcels -> near zero.
        %
        %  COMPONENT 3 – Forward progress (+w_progress x total_dy)
        %    Proportional to the sum of positive y-displacements this step.
        %    Prevents the agent from stalling parcels to avoid collisions.
        %
        %  COMPONENT 4 – Collision penalty (+w_collision x n_collisions)
        %    Penalises each pairwise overlap resolved in this timestep.
        %    Drives the agent to keep parcels separated.
        %
        %  COMPONENT 5 – Idle penalty (+w_idle)
        %    Small negative reward when no parcel is actively on the AMS,
        %    encouraging faster throughput.

            Reward = 0;
            l_AMS  = this.d_AMS * this.n_j_AMS;
            y_exit = this.d_AMS * this.n_i_AMS;

            active = false(this.max_gen_boxes, 1);
            for ii = 1:this.n_boxes_tot
                active(ii) = (this.x_boxes(ii) >= 0) && ...
                             (this.y_boxes(ii) < y_exit) && ...
                             (this.pack_exited(ii) == 0);
            end
            n_active = sum(active);

            % 1. Throughput -----------------------------------------------
            for ii = 1:this.n_boxes_tot
                if this.pack_exited(ii) == 1 && this.prev_pack_exited(ii) == 0
                    Reward = Reward + this.w_throughput;
                end
            end

            % 2. Spacing ---------------------------------------------------
            active_idx = find(active);
            if numel(active_idx) >= 2
                x_act = this.x_boxes(active_idx);
                d_act = this.d_boxes(active_idx);
                [x_s, so] = sort(x_act);
                d_s  = d_act(so);
                gaps = zeros(numel(x_s)-1, 1);
                for k = 1:numel(x_s)-1
                    gaps(k) = (x_s(k+1) - d_s(k+1)/2) - (x_s(k) + d_s(k)/2);
                end
                gap_score = min(max(gaps, 0)) / l_AMS;
                Reward    = Reward + this.w_spacing * gap_score;
            elseif numel(active_idx) == 1
                Reward = Reward + this.w_spacing;  % perfect singulation
            end

            % 3. Forward progress -----------------------------------------
            progress = 0;
            for ii = 1:this.n_boxes_tot
                if active(ii)
                    dy = this.y_boxes(ii) - this.prev_y_boxes(ii);
                    if dy > 0
                        progress = progress + dy / y_exit;
                    end
                end
            end
            Reward = Reward + this.w_progress * progress;

            % 4. Collision penalty ----------------------------------------
            Reward = Reward + this.w_collision * this.n_collisions;

            % 5. Idle penalty ---------------------------------------------
            if n_active == 0
                Reward = Reward + this.w_idle;
            end
        end

        function set.State(this, state)
            validateattributes(state, {'numeric'}, ...
                {'finite','real','vector','numel',100}, '', 'State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end

    end

    %% ================================================================
    %  Protected / private
    %% ================================================================
    methods (Access = protected)
        function envUpdatedCallback(~)
        end
    end

    methods (Access = private)

        function obs = buildObservation(this)
        %BUILDOBSERVATION  100-element observation vector (all in [0,1]).

            obs    = zeros(100, 1);
            l_AMS  = this.d_AMS * this.n_j_AMS;
            y_exit = this.d_AMS * this.n_i_AMS;
            max_d  = 2.0 * this.d_AMS;
            v_max  = this.upperlim_v;

            for k = 1:this.max_gen_boxes
                base = 10*(k-1);

                if k > this.n_boxes_tot || this.x_boxes(k) < 0
                    obs(base+1 : base+10) = 0;
                    continue
                end

                exited = double(this.pack_exited(k));
                active = double(~logical(exited) && this.y_boxes(k) < y_exit);

                x_n = max(0, min(1, this.x_boxes(k) / l_AMS));
                y_n = max(0, min(1, this.y_boxes(k) / y_exit));
                d_n = max(0, min(1, this.d_boxes(k) / max_d));

                ct = this.cont;
                if ct > 1
                    vx_r = (this.x_boxes(k) - this.x_boxes_vect(k, ct-1)) / this.dt;
                    vy_r = (this.y_boxes(k) - this.y_boxes_vect(k, ct-1)) / this.dt;
                else
                    vx_r = 0;  vy_r = 0;
                end
                vx_n = max(0, min(1, (vx_r + v_max) / (2*v_max)));
                vy_n = max(0, min(1, (vy_r + v_max) / (2*v_max)));

                if numel(this.i_box) >= k && numel(this.j_box) >= k
                    i_n = max(0, min(1, this.i_box(k) / this.n_i_AMS));
                    j_n = max(0, min(1, this.j_box(k) / this.n_j_AMS));
                else
                    i_n = 0;  j_n = 0;
                end

                sp_min = 1.0;
                for m = 1:this.n_boxes_tot
                    if m ~= k && this.x_boxes(m) >= 0
                        dist = sqrt((this.x_boxes(k)-this.x_boxes(m))^2 + ...
                                    (this.y_boxes(k)-this.y_boxes(m))^2) - ...
                               (this.d_boxes(k)/2 + this.d_boxes(m)/2);
                        sp_min = min(sp_min, max(dist, 0) / l_AMS);
                    end
                end

                obs(base+1)  = active;
                obs(base+2)  = x_n;
                obs(base+3)  = y_n;
                obs(base+4)  = d_n;
                obs(base+5)  = vx_n;
                obs(base+6)  = vy_n;
                obs(base+7)  = j_n;
                obs(base+8)  = i_n;
                obs(base+9)  = exited;
                obs(base+10) = sp_min;
            end
        end

    end
end
