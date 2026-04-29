classdef RL_environment1 < rl.env.MATLABEnvironment
    %AMS_RL: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        v_treadmill = 0.6; % m/s

        d_AMS = 0.2; % m - size of the AMS
        n_i_AMS = 5; % numeber of AMS along y
        n_j_AMS = 5; % number of AMS along x
        n_actions_art = 2; % number of actions of each AMS

        upperlim_rot = pi*45/180; % upper limit rotation for AMS action
        upperlim_v = 2.2; % max velocity for AMS
        lowerlim_rot = - pi*45/180; % lower limit rotation for AMS action
        lowerlim_v = 0.5; % min velocity for AMS

        vy_boxes = [];
        x_boxes_prec = [];
        y_boxes_prec = [];

        index_AMS = [];

        LastAction = zeros(50,1);

        i_box = [];
        j_box = [];

        n_boxes_tot_vect = [];
        n_boxes_tot = 0;
        max_gen_boxes = 10;
        cont_new_box = 0;

        d_boxes = zeros(10,1);
        x_boxes = zeros(10,1);
        y_boxes = zeros(10,1);
        x_boxes_vect = zeros(10000,10);
        y_boxes_vect = zeros(10000,10);
        cont = 0;
        pack_exited = zeros(10,1);
        exit_order = zeros(10,1);
        index_exit = 0;

        toll_contatto = 0.005;

        dt = 0.01; % s - timestep for simulation

        time = 0; % s - simulation time
    end

    properties (Hidden)
        % Flags for visualization
        VisualizeAnimation = false
        VisualizeActions = false
        VisualizeStates = false        
    end
    
    properties
        % Initialize system state [on1,x1,y1,on2,x2,y2,...,on25,x25,y25]'
        State = zeros(50,1) % Deve essere uguale alla dimensione di 
                % 'ObservationInfo', definito dopo. Se passiamo solo la
                % posizione dei pacchi (30 valori) ==> 30 anzi che 100
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    properties (Transient, Access = private)
        Visualizer = []
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = RL_environment1()
            % Initialize Observation settings

%---------- ObservationInfo = ???; % states: to be defined ----------
            % Definiamo 3 osservazioni per ogni pacco (x, y, diametro) per un totale di 30
            %OCCHIO: Potresti voler definire 5 osservazioni per ogni pacco
            %   (x, y, diametro, x_prec, y_prec)


            numObservations = 50; 
            ObservationInfo = rlNumericSpec([numObservations 1]);
            ObservationInfo.Name = 'Parcel States';
            ObservationInfo.Description = 'x, y, x_prec, y_prec and diameter for each of the 10 parcels';

            % Initialize Action settings
            n_actions = 5*5*2;
            ActionInfo = rlNumericSpec([n_actions 1]);
            ActionInfo.Name = 'AMS Action';
            ActionInfo.Description = 'r1, v1, r2, v2, ...';
            ActionInfo.LowerLimit = zeros(n_actions,1);
            ActionInfo.UpperLimit = zeros(n_actions,1);

            for ii=1:2:n_actions

                ActionInfo.UpperLimit(ii) = pi*45/180;
                ActionInfo.UpperLimit(ii+1) = 2.2;
                ActionInfo.LowerLimit(ii) = - pi*45/180;
                ActionInfo.LowerLimit(ii+1) = 0.5;
                
            end
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,Info] = step(this,Action)
            Info = [];

            this.time = this.time + this.dt;

            this.cont = this.cont + 1;

            l_AMS_matrix = this.d_AMS*this.n_j_AMS; % m

            ActLimUp = zeros(50,1);
            ActLimLow = zeros(50,1);

            for ii=1:2:50
                ActLimUp(ii) = this.upperlim_rot;
                ActLimUp(ii+1) = this.upperlim_v;
                ActLimLow(ii) = this.lowerlim_rot;
                ActLimLow(ii+1) = this.lowerlim_v;
            end

            % Actions are normalized [0-1]
            % De-normalizing the actions
            AMS_actions = ActLimLow + (1 + Action) .* (ActLimUp - ActLimLow)./2;
            for ii=1:50
                AMS_actions(ii) = max(ActLimLow(ii),min(ActLimUp(ii),AMS_actions(ii)));
            end

            this.LastAction = AMS_actions;

            v_AMS = zeros(5,5);
            rotation_AMS = zeros(5,5);

            for ii = 1:this.n_i_AMS
                for jj = 1:this.n_j_AMS
                    % rotational actions
                    rotation_AMS(ii,jj) = AMS_actions(jj*2-1+(ii-1)*10);
                    % velocity actions
                    v_AMS(ii,jj) = AMS_actions(jj*2+(ii-1)*10);
                end
            end

            % new boxes generation each 0.75 s. 2 boxes are generated.

            if mod(round(this.time,2),0.75) == 0 && this.n_boxes_tot<this.max_gen_boxes && this.time>0.25
                
                this.cont_new_box = this.cont_new_box + 1;
                
                gen_n_boxes = 2;
                if gen_n_boxes>2
                    gen_n_boxes=2;
                end

                while this.n_boxes_tot+gen_n_boxes>this.max_gen_boxes
                    gen_n_boxes = gen_n_boxes-1;
                end

                this.n_boxes_tot = this.n_boxes_tot+gen_n_boxes;
                this.n_boxes_tot_vect(this.cont_new_box) = gen_n_boxes;

                for ii=1:gen_n_boxes

                    if gen_n_boxes>1
                        if ii == 1
                            this.x_boxes(this.n_boxes_tot-1) = this.d_boxes(this.n_boxes_tot-1)*0.5 + (this.d_AMS*2.5-this.d_boxes(this.n_boxes_tot-1)*0.5)*rand(1);
                            this.y_boxes(this.n_boxes_tot-1) = 0.001 + rand(1)*0.01;
                            this.x_boxes_prec(this.n_boxes_tot-1) = this.x_boxes(this.n_boxes_tot-1);
                            this.y_boxes_prec(this.n_boxes_tot-1) = this.y_boxes(this.n_boxes_tot-1);
                        else
                            this.x_boxes(this.n_boxes_tot) = this.d_AMS*2.5+this.d_boxes(this.n_boxes_tot)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(this.n_boxes_tot)*0.5))*rand(1);
                            this.y_boxes(this.n_boxes_tot) = 0.001 + rand(1)*0.05;
                            if abs(this.x_boxes(this.n_boxes_tot)-this.x_boxes(this.n_boxes_tot-1)) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                                this.x_boxes(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot-1) + this.d_boxes(this.n_boxes_tot)/2 + this.d_boxes(this.n_boxes_tot-1)/2 + 0.1;
                            end
                            if this.x_boxes(this.n_boxes_tot)-this.d_boxes(this.n_boxes_tot)/2<0
                                this.x_boxes(this.n_boxes_tot) = this.d_boxes(this.n_boxes_tot)/2;
                            elseif this.x_boxes(this.n_boxes_tot)+this.d_boxes(this.n_boxes_tot)/2>l_AMS_matrix
                                this.x_boxes(this.n_boxes_tot) = l_AMS_matrix-this.d_boxes(this.n_boxes_tot)/2;
                            end
                            if abs(this.x_boxes(this.n_boxes_tot)-this.x_boxes(this.n_boxes_tot-1)) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                                this.x_boxes(this.n_boxes_tot-1) = this.x_boxes(this.n_boxes_tot) - (this.d_boxes(this.n_boxes_tot)/2 + this.d_boxes(this.n_boxes_tot-1)/2 + 0.1);
                            end
                            this.x_boxes_prec(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                            this.y_boxes_prec(this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                        end
                    else
                        this.x_boxes(this.n_boxes_tot) = this.d_boxes(this.n_boxes_tot)*0.55 + (l_AMS_matrix-this.d_boxes(this.n_boxes_tot)*0.55)*rand(1);
                        this.y_boxes(this.n_boxes_tot) = 0.001 + rand(1)*0.01;
                        this.x_boxes_prec(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_prec(this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                        this.x_boxes_vect(this.n_boxes_tot,this.cont-1) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_vect(this.n_boxes_tot,this.cont-1) = this.y_boxes(this.n_boxes_tot);
                    end

                end

            end

            for ii=this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end

            act_AMS = zeros(25,1);
            this.index_AMS = [];

            % checking collisions

            for ii=1:this.n_boxes_tot

                if this.x_boxes(ii)-this.d_boxes(ii)/2<0
                    this.x_boxes(ii) = this.d_boxes(ii)/2;
                elseif this.x_boxes(ii)+this.d_boxes(ii)/2>l_AMS_matrix
                    this.x_boxes(ii) = l_AMS_matrix-this.d_boxes(ii)/2;
                end

                % identifying on which AMS the boxes are (geometrical baricenter)

                if this.y_boxes(ii)<=this.d_AMS
                    this.i_box(ii) = 1;
                elseif this.y_boxes(ii)<=this.d_AMS*2
                    this.i_box(ii) = 2;
                elseif this.y_boxes(ii)<=this.d_AMS*3
                    this.i_box(ii) = 3;
                elseif this.y_boxes(ii)<=this.d_AMS*4
                    this.i_box(ii) = 4;
                elseif this.y_boxes(ii)<=this.d_AMS*5
                    this.i_box(ii) = 5;
                else
                    this.i_box(ii) = 6;
                end

                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                elseif this.x_boxes(ii)<=this.d_AMS
                    this.j_box(ii) = 1;
                elseif this.x_boxes(ii)<=this.d_AMS*2
                    this.j_box(ii) = 2;
                elseif this.x_boxes(ii)<=this.d_AMS*3
                    this.j_box(ii) = 3;
                elseif this.x_boxes(ii)<=this.d_AMS*4
                    this.j_box(ii) = 4;
                else
                    this.j_box(ii) = 5;
                end

                % packages kinematics

                if this.i_box(ii) < 6
                    this.x_boxes(ii) = this.x_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*sin(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.y_boxes(ii) = this.y_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.vy_boxes(ii) = v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                else
                    this.x_boxes(ii) = this.x_boxes(ii);
                    this.y_boxes(ii) = this.y_boxes(ii) + this.v_treadmill*this.dt;
                    this.vy_boxes(ii) = this.v_treadmill;
                end

                % collisions

                if ii>1 && this.cont>1
                    for jj=ii:-1:2
                        if abs(this.x_boxes(ii)-this.x_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 + 0.001 ...
                                && abs(this.x_boxes_vect(ii,this.cont-1)-this.x_boxes_vect(jj-1,this.cont-1)) >= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.y_boxes(ii)-this.y_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.y_boxes_vect(ii,this.cont-1)-this.y_boxes_vect(jj-1,this.cont-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2
                            if (this.x_boxes(ii)-this.d_boxes(ii)/2 <= this.x_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.x_boxes(ii)-this.d_boxes(ii)/2>this.x_boxes(jj-1)-this.d_boxes(jj-1)/2)
                                penetrazione_x = (this.x_boxes(jj-1)+this.d_boxes(jj-1)/2) - (this.x_boxes(ii)-this.d_boxes(ii)/2);
                                this.x_boxes(ii) = this.x_boxes(ii) + penetrazione_x/2 + this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) - penetrazione_x/2 - this.toll_contatto;
                            elseif (this.x_boxes(ii)+this.d_boxes(ii)/2 >= this.x_boxes(jj-1)-this.d_boxes(jj-1)/2 && this.x_boxes(ii)+this.d_boxes(ii)/2<this.x_boxes(jj-1)+this.d_boxes(jj-1)/2)
                                penetrazione_x = (this.x_boxes(ii)+this.d_boxes(ii)/2) - (this.x_boxes(jj-1)-this.d_boxes(jj-1)/2);
                                this.x_boxes(ii) = this.x_boxes(ii) - penetrazione_x/2 - this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) + penetrazione_x/2 + this.toll_contatto;
                            end
                        elseif abs(this.y_boxes(ii)-this.y_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 + 0.001 ...
                                && abs(this.y_boxes_vect(ii,this.cont-1)-this.y_boxes_vect(jj-1,this.cont-1)) >= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.x_boxes(ii)-this.x_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                            if (this.y_boxes(ii)-this.d_boxes(ii)/2 <= this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.y_boxes(ii)-this.d_boxes(ii)/2>this.y_boxes(jj-1)-this.d_boxes(jj-1)/2) % || (this.y_boxes(jj-1)-this.d_boxes(jj-1)/2 <= this.y_boxes(ii)+this.d_boxes(ii)/2 && this.y_boxes(jj-1)-this.d_boxes(jj-1)/2>this.y_boxes(ii)-this.d_boxes(ii)/2)
                                penetrazione_y = (this.y_boxes(jj-1)+this.d_boxes(jj-1)/2)-(this.y_boxes(ii)-this.d_boxes(ii)/2);
                                this.y_boxes(ii) = this.y_boxes(ii) + penetrazione_y/2 + this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) - penetrazione_y/2 - this.toll_contatto;
                            elseif (this.y_boxes(ii)+this.d_boxes(ii)/2 < this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.y_boxes(ii)+this.d_boxes(ii)/2 >= this.y_boxes(jj-1)-this.d_boxes(jj-1)/2) % || (this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 < this.y_boxes(ii)+this.d_boxes(ii)/2 && this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 >= this.y_boxes(ii)-this.d_boxes(ii)/2)
                                penetrazione_y = (this.y_boxes(ii)+this.d_boxes(ii)/2) - (this.y_boxes(jj-1)-this.d_boxes(jj-1)/2);
                                this.y_boxes(ii) = this.y_boxes(ii) - penetrazione_y/2  - this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) + penetrazione_y/2 + this.toll_contatto;
                            end
                        end
                    end
                end
                
                if this.i_box(ii)<6 && this.j_box(ii)<6
                    act_AMS(this.j_box(ii)+(this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii)+(this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end
            end

            % Observation: observations for the RL to be defined
            % Costruiamo il vettore delle osservazioni (50 elementi)
            % Ordine: [x_att, y_att, x_prec, y_prec, diametro] x 10 pacchi
            Observation = [this.x_boxes;       ... % 1-10: X attuali
                this.y_boxes;   ... % 11-20: Y attuali
                this.x_boxes_prec;  ... % 21-30: X precedenti
                this.y_boxes_prec;  ... % 31-40: Y precedenti
                this.d_boxes];          % 41-50: Diametri

            for ii=1:this.max_gen_boxes
                if ii<=this.n_boxes_tot
                    this.x_boxes_vect(ii,this.cont) = this.x_boxes(ii);
                    this.y_boxes_vect(ii,this.cont) = this.y_boxes(ii);
                    this.x_boxes_prec(ii) = this.x_boxes(ii);
                    this.y_boxes_prec(ii) = this.y_boxes(ii);
                else
                    this.x_boxes_vect(ii,this.cont) = -1;
                    this.y_boxes_vect(ii,this.cont) = -1;
                end
            end

            % Update system states
            this.State = Observation;

            cont_exit = 0;

            % Check terminal condition
            for ii=1:this.n_boxes_tot
                if this.y_boxes(ii)>this.d_AMS*5
                    cont_exit = cont_exit + 1;
                    if this.pack_exited(ii) == 0
                        this.pack_exited(ii) = 1;
                        this.index_exit = this.index_exit + 1;
                        this.exit_order(this.index_exit) = ii;
                    end
                end
            end

            if cont_exit==this.max_gen_boxes
                IsDone = true;
            else
                IsDone = false;
            end

            this.IsDone = IsDone;
            
            % Get reward
            Reward = getReward(this,AMS_actions);
            
            %% OPTIONAL
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        % for each episod
        function InitialObservation = reset(this)

            this.index_exit = 0;
            this.pack_exited = zeros(10,1); % is package exited the AMS or not?
            this.exit_order = zeros(10,1); % packages ordered by exit

            this.cont_new_box = 1;

            this.n_boxes_tot_vect = 2;
            this.n_boxes_tot = this.n_boxes_tot_vect;

            this.time = 0;

            this.vy_boxes = zeros(2,1);

            this.cont = 0;

            max_d_box = 2.*this.d_AMS; % m
            min_d_box = 0.25*this.d_AMS; % m

            l_AMS_matrix = this.d_AMS*this.n_j_AMS; % m

            % generating initial boxes (2)
            for ii=1:this.max_gen_boxes
                this.d_boxes(ii) = min_d_box + (max_d_box-min_d_box)*rand(1);
            end

            for ii=1:2
                if ii == 1
                    x1 = this.d_boxes(1)*0.5 + (this.d_AMS*2.5-this.d_boxes(1)*0.5)*rand(1);
                else
                    x2 = this.d_AMS*2.5+this.d_boxes(2)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(2)*0.5))*rand(1);
                    if abs(x2-x1) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                        x2 = x1 + this.d_boxes(2)/2 + this.d_boxes(1)/2 + 0.1;
                    end
                    if x2-this.d_boxes(2)/2<0
                        x2 = this.d_boxes(2)/2;
                    elseif x2+this.d_boxes(2)/2>l_AMS_matrix
                        x2 = l_AMS_matrix-this.d_boxes(2)/2;
                    end
                    if abs(x2-x1) <= this.d_boxes(2)/2+this.d_boxes(1)/2
                        x1 = x2 - (this.d_boxes(2)/2 + this.d_boxes(1)/2 + 0.1);
                    end
                end

                y1 = 0.001 + rand(1)*0.01;
                y2 = 0.001 + rand(1)*0.05;
            end

            this.x_boxes(1) = x1;
            this.y_boxes(1) = y1;
            this.x_boxes(2) = x2;
            this.y_boxes(2) = y2;

            this.x_boxes_prec(1) = x1;
            this.x_boxes_prec(2) = x2;
            this.y_boxes_prec(1) = y1;
            this.y_boxes_prec(2) = y2;

            for ii=this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = 0;
                this.y_boxes(ii) = 0;
            end

            act_AMS = zeros(25,1);
            this.index_AMS = zeros(2,1);

            % identifying on which AMS the boxes are (geometrical
            % baricenter)

            for ii=1:this.n_boxes_tot

                if this.y_boxes(ii)<=this.d_AMS
                    this.i_box(ii) = 1;
                elseif this.y_boxes(ii)<=this.d_AMS*2
                    this.i_box(ii) = 2;
                elseif this.y_boxes(ii)<=this.d_AMS*3
                    this.i_box(ii) = 3;
                elseif this.y_boxes(ii)<=this.d_AMS*4
                    this.i_box(ii) = 4;
                elseif this.y_boxes(ii)<=this.d_AMS*5
                    this.i_box(ii) = 5;
                else
                    this.i_box(ii) = 6;
                end

                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                elseif this.x_boxes(ii)<=this.d_AMS
                    this.j_box(ii) = 1;
                elseif this.x_boxes(ii)<=this.d_AMS*2
                    this.j_box(ii) = 2;
                elseif this.x_boxes(ii)<=this.d_AMS*3
                    this.j_box(ii) = 3;
                elseif this.x_boxes(ii)<=this.d_AMS*4
                    this.j_box(ii) = 4;
                else
                    this.j_box(ii) = 5;
                end

                if this.i_box(ii)<6 && this.j_box(ii)<6
                    act_AMS(this.j_box(ii)+(this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii)+(this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end

            end

            % InitialObservation: initial observation for the RL to be
            % defined
            % Costruiamo il vettore delle osservazioni iniziali (50 elementi)
            % Deve seguire lo stesso ordine logico usato nel metodo step
            InitialObservation = [this.x_boxes(:);       ... % 1-10: X iniziali
                this.y_boxes(:);       ... % 11-20: Y iniziali
                this.x_boxes_prec(:);  ... % 21-30: X precedenti (uguali alle attuali al reset)
                this.y_boxes_prec(:);  ... % 31-40: Y precedenti (uguali alle attuali al reset)
                this.d_boxes(:)];          % 41-50: Diametri dei pacchi
            this.State = InitialObservation;
            
            %% OPTIONAL
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);

        end
    end


    %% Optional Methods (set methods' attributes accordingly)
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
            % Reset Visualizations
            this.VisualizeAnimation = true;
            this.VisualizeActions = false;
            this.VisualizeStates = false;
        end

        % Reward function
        function Reward = getReward(this, AMS_actions)
            % 1. Inizializzazione
            Reward = 0;
    
            % 2. Penalità collisioni (usa il loop scritto sopra)
            penalty_collision = 0;
            for ii = 1:this.n_boxes_tot
                for jj = ii+1:this.n_boxes_tot
                    dist = sqrt((this.x_boxes(ii)-this.x_boxes(jj))^2 + (this.y_boxes(ii)-this.y_boxes(jj))^2);
                    if dist < (this.d_boxes(ii) + this.d_boxes(jj))/2 + this.toll_contatto
                        penalty_collision = penalty_collision - 100; % Penalità pesante
                    end
                end
            end
            Reward = Reward + penalty_collision;
            % 3. Premio per uscita pacchi (singulation)
            % Se un pacco è appena uscito, dai un premio
            Reward = Reward + (this.index_exit * 10); 

            % 4. Penalità per eccessivo stazionamento
            Reward = Reward - 0.5; 
    
            % 5. (Opzionale) Penalità se i pacchi escono troppo vicini
            % Puoi usare this.exit_order per vedere quali sono usciti per ultimi
        end
        
        %% OPTIONAL
        % (optional) Properties validation through set methods
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',50},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end
        
    end
    
    methods (Access = protected)
        %% OPTIONAL
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
            disp(['Dimensione stato attuale: ', num2cell(numel(this.State))]);
        end
    end
end
