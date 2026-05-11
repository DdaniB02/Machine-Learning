classdef RL_environment1 < rl.env.MATLABEnvironment
<<<<<<< Updated upstream
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
=======
    %RL_environment1  Ambiente RL per smistamento pacchi su matrice AMS 5x5.
    %
    % Paper di riferimento:
    %   "Optimizing Parcels Sorting Through Reinforcement Learning
    %    for Intralogistics" – Roveda et al., ECAI 2025.
    %
    % =========================================================================
    % CHANGELOG
    % =========================================================================
    % [FIX-1]  Cinematica corretta: vx=av·cos(ar), vy=av·sin(ar)
    % [FIX-2]  Collisioni solo laterali (asse X); overlap frontale permesso
    % [FIX-3]  Stato per cella AMS, dim 100 (4×25), indice k=i+j*5
    % [NEW-1]  Reward: MEDIA dei local_score invece di min
    % [NEW-2]  Curriculum a 3 fasi: n_pacchi, diametri, azioni
    % [NEW-3]  Action masking: celle lontane dai pacchi fissate a default
    % [NEW-4]  Normalizzazione osservazioni: statica [0,1] (più stabile con stato sparse)
    % [NEW-5]  v_treadmill = 1.9 m/s
    % [NEW-6]  plot() usa AMSVisualizer_simple di default (più veloce)

    %% =====================================================================
    properties
        % --- Fisica AMS ---
        v_treadmill  = 1.9;          % m/s  [NEW-5]
        d_AMS        = 0.2;          % m
        n_i_AMS      = 5;
        n_j_AMS      = 5;
        n_actions_art = 2;

        upperlim_rot =  pi*45/180;
        upperlim_v   =  2.2;
        lowerlim_rot = -pi*45/180;
        lowerlim_v   =  0.5;

        % --- Stato pacchi ---
        x_boxes      = zeros(10,1);
        y_boxes      = zeros(10,1);
        x_boxes_prec = zeros(10,1);
        y_boxes_prec = zeros(10,1);
        vx_boxes     = zeros(10,1);
        vy_boxes     = zeros(10,1);
        d_boxes      = zeros(10,1);

        % --- Indici AMS (0-based) ---
        i_box     = zeros(10,1);
        j_box     = zeros(10,1);
        index_AMS = zeros(10,1);

        LastAction = zeros(50,1);

        % --- Gestione pacchi ---
        n_boxes_tot   = 0;
        max_gen_boxes = 10;
        cont_new_box  = 0;
        next_gen_step = 40;

        % --- Uscite ---
        pack_exited     = zeros(10,1);
        exit_order      = zeros(10,1);
        index_exit      = 0;
        index_exit_prev = 0;

        % --- Area di reward ---
        reward_area_length = 1.0;    % m dopo yExit
        in_reward_area     = zeros(10,1);

        % --- Curriculum [NEW-2] ---
        % level 0→1 in 500 episodi (step=0.002/ep)
        % Fase 1 [0.00-0.40): 2 pacchi, diam fisso, action mask attiva
        % Fase 2 [0.40-0.80): pacchi crescenti, diam interpolato
        % Fase 3 [0.80-1.00]: configurazione piena
        curriculum_level = 0.0;
        curriculum_step  = 0.002;

        % --- Collisioni ---
        toll_contatto      = 0.002;
        coeff_restituzione = 0.1;

        % --- Tempo ---
        dt   = 0.01;
        time = 0;
        cont = 0;
>>>>>>> Stashed changes
    end

    properties (Hidden)
        % Flags for visualization
        VisualizeAnimation = false
        VisualizeActions = false
        VisualizeStates = false        
    end
    
    properties
<<<<<<< Updated upstream
        % Initialize system state [on1,x1,y1,on2,x2,y2,...,on25,x25,y25]'
        State = zeros(50,1) % Deve essere uguale alla dimensione di 
                % 'ObservationInfo', definito dopo. Se passiamo solo la
                % posizione dei pacchi (30 valori) ==> 30 anzi che 100
=======
        State = zeros(100,1)
>>>>>>> Stashed changes
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    properties (Transient, Access = private)
        Visualizer = []
    end

<<<<<<< Updated upstream
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
                
=======
    %% =====================================================================
    %  Metodi necessari
    %% =====================================================================
    methods

        function this = RL_environment1()
            numObs = 4*5*5;   % 100
            ObservationInfo = rlNumericSpec([numObs 1]);
            ObservationInfo.Name = 'AMS Cell States (norm)';

            n_act = 5*5*2;    % 50
            ActionInfo = rlNumericSpec([n_act 1]);
            ActionInfo.Name = 'AMS Actions';
            lim_low = zeros(n_act,1);
            lim_up  = zeros(n_act,1);
            for ii = 1:2:n_act
                lim_up(ii)    =  pi*45/180;
                lim_up(ii+1)  =  2.2;
                lim_low(ii)   = -pi*45/180;
                lim_low(ii+1) =  0.5;
>>>>>>> Stashed changes
            end
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
        end
<<<<<<< Updated upstream
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,Info] = step(this,Action)
=======

        % -----------------------------------------------------------------
        function [Observation, Reward, IsDone, Info] = step(this, Action)
>>>>>>> Stashed changes
            Info = [];
            this.time = this.time + this.dt;

            this.cont = this.cont + 1;

<<<<<<< Updated upstream
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
=======
            L = this.d_AMS * this.n_j_AMS;
            H = this.d_AMS * this.n_i_AMS;

            % De-normalizzazione azioni [-1,1] → fisici
            lim_up  = repmat([this.upperlim_rot; this.upperlim_v], 25, 1);
            lim_low = repmat([this.lowerlim_rot; this.lowerlim_v], 25, 1);
            AMS_actions = lim_low + (1+Action).*(lim_up-lim_low)./2;
            AMS_actions = max(lim_low, min(lim_up, AMS_actions));

            % Action masking nelle fasi iniziali [NEW-3]
            if this.curriculum_level < 0.8
                AMS_actions = this.applyActionMask(AMS_actions);
            end
            this.LastAction = AMS_actions;

            % Matrici 5×5 rotazione/velocità
            rotation_AMS = zeros(5,5);
            v_AMS        = zeros(5,5);
            for i0 = 0:4
                for j0 = 0:4
                    k = i0 + j0*5;
                    rotation_AMS(i0+1,j0+1) = AMS_actions(2*k+1);
                    v_AMS(i0+1,j0+1)        = AMS_actions(2*k+2);
                end
            end

            % Generazione pacchi
            if this.cont >= this.next_gen_step && ...
                    this.n_boxes_tot < this.maxBoxesForLevel()
                this = generaPacki(this, L);
                this.next_gen_step = this.cont + 40 + round(rand()*60);
            end
            for ii = this.n_boxes_tot+1:this.max_gen_boxes
>>>>>>> Stashed changes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end

<<<<<<< Updated upstream
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
=======
            % Salva posizioni precedenti
            this.x_boxes_prec(1:this.n_boxes_tot) = this.x_boxes(1:this.n_boxes_tot);
            this.y_boxes_prec(1:this.n_boxes_tot) = this.y_boxes(1:this.n_boxes_tot);

            % Cinematica: ar=0 → dritto avanti (+Y)
            %   vx = av · sin(ar)   (deviazione laterale)
            %   vy = av · cos(ar)   (avanzamento principale)
            % Con ar=0: vx=0, vy=av → pacco avanza sempre in Y ✓
            for ii = 1:this.n_boxes_tot
                [i_b, j_b] = getAMSIndex(this, this.x_boxes(ii), this.y_boxes(ii));
                this.i_box(ii) = i_b;
                this.j_box(ii) = j_b;
                if i_b <= 4
                    ar = rotation_AMS(i_b+1, j_b+1);
                    av = v_AMS(i_b+1, j_b+1);
                    vx = av * sin(ar);   % deviazione laterale
                    vy = av * cos(ar);   % avanzamento in Y (max quando ar=0)
                else
                    vx = 0;
                    vy = this.v_treadmill;
                end
                this.vx_boxes(ii) = vx;
                this.vy_boxes(ii) = vy;
                this.x_boxes(ii)  = this.x_boxes(ii) + vx*this.dt;
                this.y_boxes(ii)  = this.y_boxes(ii) + vy*this.dt;
            end

            % Collisioni laterali [FIX-2]
            this = risolviColisioniLaterali(this);

            % Clamp bordi laterali
            for ii = 1:this.n_boxes_tot
                r = this.d_boxes(ii)/2;
                this.x_boxes(ii) = max(r, min(L-r, this.x_boxes(ii)));
            end

            % Aggiorna indici AMS
            for ii = 1:this.n_boxes_tot
                [i_b, j_b] = getAMSIndex(this, this.x_boxes(ii), this.y_boxes(ii));
                this.i_box(ii) = i_b;
                this.j_box(ii) = j_b;
                if i_b <= 4 && j_b <= 4
                    this.index_AMS(ii) = i_b + j_b*5;
>>>>>>> Stashed changes
                else
                    this.index_AMS(ii) = 25;
                end
            end

<<<<<<< Updated upstream
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

=======
            % Uscite e avanzamento treadmill
            y_exit_line = H;
            y_eval_line = H + this.reward_area_length;
            this.index_exit_prev = this.index_exit;
            for ii = 1:this.n_boxes_tot
                if this.x_boxes(ii) < 0, continue; end
                if this.y_boxes(ii) > y_exit_line && this.pack_exited(ii) == 0
                    this.pack_exited(ii) = 1;
                    this.index_exit = this.index_exit + 1;
                    this.exit_order(this.index_exit) = ii;
                    this.in_reward_area(ii) = 1;
                end
                if this.in_reward_area(ii)
                    this.y_boxes(ii) = this.y_boxes(ii) + this.v_treadmill*this.dt;
                    if this.y_boxes(ii) - this.d_boxes(ii)/2 > y_eval_line
                        this.in_reward_area(ii) = 0;
                    end
                end
            end

            Observation = buildObservation(this);
            this.State  = Observation;

            IsDone = (sum(this.pack_exited(1:this.n_boxes_tot)) == this.max_gen_boxes);
            this.IsDone = IsDone;

            Reward = getReward(this);

            notifyEnvUpdated(this);
        end

        % -----------------------------------------------------------------
        function InitialObservation = reset(this)
            this.index_exit      = 0;
            this.index_exit_prev = 0;
            this.pack_exited     = zeros(10,1);
            this.exit_order      = zeros(10,1);
            this.in_reward_area  = zeros(10,1);

            % Curriculum avanza di un passo [NEW-2]
            this.curriculum_level = min(1.0, this.curriculum_level + this.curriculum_step);

            this.cont_new_box = 1;
            this.n_boxes_tot  = 2;
>>>>>>> Stashed changes
            this.time = 0;

            this.vy_boxes = zeros(2,1);

            this.cont = 0;
            this.next_gen_step = 40 + round(rand()*60);





<<<<<<< Updated upstream
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
=======
            L         = this.d_AMS * this.n_j_AMS;
            max_d_box = 2.0  * this.d_AMS;
            min_d_box = 0.25 * this.d_AMS;
            d_medio   = (max_d_box + min_d_box) / 2;

            % Diametri casuali solo dalla fase 3 [NEW-2]
            diam_rw = max(0, (this.curriculum_level - 0.8) / 0.2);
            for ii = 1:this.max_gen_boxes
                d_rand = min_d_box + (max_d_box - min_d_box)*rand(1);
                this.d_boxes(ii) = (1-diam_rw)*d_medio + diam_rw*d_rand;
            end

            % Posiziona 2 pacchi iniziali
            x1 = this.d_boxes(1)*0.5 + (this.d_AMS*2.5 - this.d_boxes(1)*0.5)*rand(1);
            x2 = this.d_AMS*2.5 + this.d_boxes(2)*0.5 + ...
                 (L - (this.d_AMS*2.5 + this.d_boxes(2)*0.5))*rand(1);
            gap_min = this.d_boxes(1)/2 + this.d_boxes(2)/2 + this.toll_contatto;
            if abs(x2-x1) < gap_min, x2 = x1 + gap_min + 0.05; end
            x2 = max(this.d_boxes(2)/2, min(L-this.d_boxes(2)/2, x2));
            if abs(x2-x1) < gap_min, x1 = x2 - gap_min - 0.05; end
            x1 = max(this.d_boxes(1)/2, min(L-this.d_boxes(1)/2, x1));

            y1 = 0.001 + rand(1)*0.01;
            y2 = 0.001 + rand(1)*0.05;

            this.x_boxes(1) = x1;  this.y_boxes(1) = y1;
            this.x_boxes(2) = x2;  this.y_boxes(2) = y2;
            this.x_boxes_prec(1) = x1; this.y_boxes_prec(1) = y1;
            this.x_boxes_prec(2) = x2; this.y_boxes_prec(2) = y2;
            this.vx_boxes = zeros(10,1);
            this.vy_boxes = zeros(10,1);

            for ii = 3:this.max_gen_boxes
                this.x_boxes(ii) = -1; this.y_boxes(ii) = -1;
                this.x_boxes_prec(ii) = -1; this.y_boxes_prec(ii) = -1;
            end

            this.index_AMS = zeros(10,1);
            for ii = 1:this.n_boxes_tot
                [i_b, j_b] = getAMSIndex(this, this.x_boxes(ii), this.y_boxes(ii));
                this.i_box(ii) = i_b;
                this.j_box(ii) = j_b;
                if i_b <= 4 && j_b <= 4
                    this.index_AMS(ii) = i_b + j_b*5;
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream

    %% Optional Methods (set methods' attributes accordingly)
    methods  
=======
    end % methods necessari

    %% =====================================================================
    %  Metodi pubblici opzionali
    %% =====================================================================
    methods
>>>>>>> Stashed changes

        % plot(env)         → AMSVisualizer_simple  (default, leggero) [NEW-6]
        % plot(env,'full')  → AMSVisualizer          (con frecce AMS)
        function varargout = plot(this, mode)
            if nargin < 2, mode = 'simple'; end
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                if strcmpi(mode,'full')
                    this.Visualizer = AMSVisualizer(this);
                else
                    this.Visualizer = AMSVisualizer_simple(this);
                end
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

<<<<<<< Updated upstream
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
=======
        % -----------------------------------------------------------------
        % Reward — tutte le componenti normalizzate in [0,1] per step
        %
        % Obiettivo: segnale denso e varianza bassa → critic stabile.
        % La reward totale per step è nell'ordine [0, ~1].
        % Su 1000 step/episodio: reward episodio attesa ≈ [0, ~1000].
        %
        % COMPONENTI:
        %
        % A) AVANZAMENTO IN Y  (peso 0.4)
        %    r_Y ∈ [0,1]: progresso medio in Y dei pacchi in griglia,
        %    normalizzato su av_max*dt (max avanzamento possibile per step).
        %    Segnale denso fin dal primo step → guida l'agente a non restare fermo.
        %
        % B) SEPARAZIONE LATERALE  (peso 0.3)
        %    r_X ∈ [0,1]: per ogni coppia di pacchi in griglia, misura
        %    quanto sono lateralmente separati rispetto alla somma dei raggi.
        %    sep_norm = tanh( |dx| / (r_i + r_j) )   → 0 se sovrapposti, 1 se distanti
        %    Media su tutte le coppie.
        %    Incentiva la separazione X durante la traversata → rompe il plateau.
        %
        % C) SINGOLAZIONE NELL'AREA DI REWARD  (peso 0.3)
        %    r_sing ∈ [0,1]: gap normalizzato tra coppie consecutive nell'area
        %    [yExit, yEval]. Usato tanh(dk / d_medio) per avere output [0,1].
        %    Bonus +0.3 se tutti gap > 0 e K ≥ 3 (aggiunta al termine C).
        %    Coerente con il KPI del visualizzatore.
        %
        % r_total = 0.4*r_Y + 0.3*r_X + 0.3*r_sing   ∈ [0, ~1] per step
        % -----------------------------------------------------------------
        function Reward = getReward(this)
            av_max = this.upperlim_v;
            L      = this.d_AMS * this.n_j_AMS;

            % --- A) Avanzamento in Y ----------------------------------------
            r_Y       = 0;
            n_in_grid = 0;
            for ii = 1:this.n_boxes_tot
                if this.x_boxes(ii) < 0 || this.pack_exited(ii), continue; end
                dy = this.y_boxes(ii) - this.y_boxes_prec(ii);
                r_Y       = r_Y + max(0, dy);
                n_in_grid = n_in_grid + 1;
            end
            if n_in_grid > 0
                r_Y = r_Y / (n_in_grid * av_max * this.dt);   % ∈ [0,1]
            end

            % --- B) Separazione laterale ------------------------------------
            r_X    = 0;
            n_pair = 0;
            for ii = 1:this.n_boxes_tot
                if this.x_boxes(ii) < 0 || this.pack_exited(ii), continue; end
                for jj = ii+1:this.n_boxes_tot
                    if this.x_boxes(jj) < 0 || this.pack_exited(jj), continue; end
                    dx         = abs(this.x_boxes(ii) - this.x_boxes(jj));
                    r_pair     = this.d_boxes(ii)/2 + this.d_boxes(jj)/2;
                    % tanh: 0 se sovrapposti, →1 se ben separati lateralmente
                    r_X   = r_X + tanh(dx / r_pair);
                    n_pair = n_pair + 1;
                end
            end
            if n_pair > 0
                r_X = r_X / n_pair;   % ∈ [0,1]
            end

            % --- C) Singolazione nell'area di reward ------------------------
            r_sing     = 0;
            idx_reward = [];
            for ii = 1:this.n_boxes_tot
                if this.in_reward_area(ii) && this.x_boxes(ii) >= 0
                    idx_reward(end+1) = ii; %#ok<AGROW>
>>>>>>> Stashed changes
                end
            end
            Reward = Reward + penalty_collision;
            % 3. Premio per uscita pacchi (singulation)
            % Se un pacco è appena uscito, dai un premio
            Reward = Reward + (this.index_exit * 10); 

<<<<<<< Updated upstream
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
=======
            K = numel(idx_reward);
            if K >= 2
                [~, ord] = sort(this.y_boxes(idx_reward));
                sorted   = idx_reward(ord);

                % Diametro medio come scala di riferimento per il gap
                d_ref  = mean(this.d_boxes(sorted));
                scores = zeros(K-1,1);
                all_pos = true;
                for kk = 1:K-1
                    pk  = sorted(kk);
                    pk1 = sorted(kk+1);
                    dk  = (this.y_boxes(pk1) - this.d_boxes(pk1)/2) - ...
                          (this.y_boxes(pk)  + this.d_boxes(pk) /2);
                    if dk <= 0, all_pos = false; end
                    % tanh normalizza il gap su scala d_ref → ∈ (-1, 1)
                    scores(kk) = tanh(dk / max(d_ref, 1e-6));
                end
                r_sing = mean(scores);   % ∈ (-1, 1], negativo se overlap
                % Bonus singolazione completa (normalizzato)
                if all_pos && K >= 3
                    r_sing = r_sing + 0.3;
                end
                % Clamp: r_sing ∈ [-1, 1.3]
                r_sing = max(-1.0, min(1.3, r_sing));
            end

            % --- Reward totale ----------------------------------------------
            % I pesi sommano a 1: la reward per step è ≈ [−0.3, 1.3]
            Reward = 0.4*r_Y + 0.3*r_X + 0.3*r_sing;
        end

        function set.State(this, state)
            validateattributes(state,{'numeric'}, ...
                {'real','vector','numel',100},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end

    end % methods pubblici

    %% =====================================================================
    %  Metodi privati
    %% =====================================================================
    methods (Access = private)

        % ----- Numero max pacchi per fase curriculum [NEW-2] ---------------
        function n = maxBoxesForLevel(this)
            if this.curriculum_level < 0.4
                n = 2;
            elseif this.curriculum_level < 0.8
                frac = (this.curriculum_level - 0.4) / 0.4;
                n = round(2 + frac*(this.max_gen_boxes - 2));
                n = max(2, min(this.max_gen_boxes, n));
            else
                n = this.max_gen_boxes;
            end
        end

        % ----- Action masking: celle non vicine ai pacchi → default [NEW-3]
        % Attivo con curriculum_level < 0.8. Le celle a distanza >1 dal
        % pacco più vicino ricevono rot=0 (dritto), vel=media.
        % Riduce lo spazio di esplorazione da 50 a ~8-18 dimensioni.
        function AMS_actions = applyActionMask(this, AMS_actions)
            active  = false(5,5);
            for ii = 1:this.n_boxes_tot
                if this.x_boxes(ii) < 0, continue; end
                i_b = this.i_box(ii);
                j_b = this.j_box(ii);
                if i_b > 4 || j_b > 4, continue; end
                for di = -1:1
                    for dj = -1:1
                        ri = i_b + di + 1;   % 1-based
                        cj = j_b + dj + 1;
                        if ri >= 1 && ri <= 5 && cj >= 1 && cj <= 5
                            active(ri, cj) = true;
                        end
                    end
                end
            end

            rot_def = 0.0;
            vel_def = (this.upperlim_v + this.lowerlim_v) / 2;

            for i0 = 0:4
                for j0 = 0:4
                    if ~active(i0+1, j0+1)
                        k = i0 + j0*5;
                        AMS_actions(2*k+1) = rot_def;
                        AMS_actions(2*k+2) = vel_def;
                    end
                end
            end
        end

        % ----- Osservazione: normalizzazione statica in [0,1] ---------------
        % Le feature raw sono già in [0,1] per costruzione (divise per L/H).
        % La normalizzazione Welford online con stato sparse (molti zeri)
        % tende a collassare le statistiche → si usa normalizzazione fissa.
        %
        % Struttura (100 elem, 4 per cella k = i+j*5):
        %   [x_k/L,  y_k/H,  w_k/L,  h_k/H]  ∈ [0,1]
        %   Se nessun pacco sulla cella k → [0, 0, 0, 0]
        function obs = buildObservation(this)
            L = this.d_AMS * this.n_j_AMS;
            H = this.d_AMS * this.n_i_AMS;

            obs = zeros(100,1);
            for ii = 1:this.n_boxes_tot
                if this.x_boxes(ii) < 0 || this.pack_exited(ii), continue; end
                i_b = this.i_box(ii);
                j_b = this.j_box(ii);
                if i_b > 4 || j_b > 4, continue; end
                k    = i_b + j_b*5;
                base = 4*k + 1;
                obs(base)   = this.x_boxes(ii) / L;
                obs(base+1) = this.y_boxes(ii) / H;
                obs(base+2) = this.d_boxes(ii) / L;
                obs(base+3) = this.d_boxes(ii) / H;
            end
            % Clamp di sicurezza
            obs = max(0, min(1, obs));
        end

        % ----- Indice cella AMS (0-based) ----------------------------------
        function [i_b, j_b] = getAMSIndex(this, x, y)
            d = this.d_AMS;
            if     y <= d,   i_b = 0;
            elseif y <= 2*d, i_b = 1;
            elseif y <= 3*d, i_b = 2;
            elseif y <= 4*d, i_b = 3;
            elseif y <= 5*d, i_b = 4;
            else,            i_b = 5;
            end
            if i_b == 5, j_b = 5; return; end
            if     x <= d,   j_b = 0;
            elseif x <= 2*d, j_b = 1;
            elseif x <= 3*d, j_b = 2;
            elseif x <= 4*d, j_b = 3;
            else,            j_b = 4;
            end
        end

        % ----- Generazione pacchi ------------------------------------------
        function this = generaPacki(this, L)
            this.cont_new_box = this.cont_new_box + 1;
            gen_n = min(2, this.maxBoxesForLevel() - this.n_boxes_tot);
            if gen_n <= 0, return; end
            this.n_boxes_tot = this.n_boxes_tot + gen_n;

            if gen_n >= 2
                idx1 = this.n_boxes_tot - 1;
                idx2 = this.n_boxes_tot;
                this.x_boxes(idx1) = this.d_boxes(idx1)*0.5 + ...
                    (this.d_AMS*2.5 - this.d_boxes(idx1)*0.5)*rand(1);
                this.y_boxes(idx1) = 0.001 + rand(1)*0.01;
                this.x_boxes(idx2) = this.d_AMS*2.5 + this.d_boxes(idx2)*0.5 + ...
                    (L - (this.d_AMS*2.5 + this.d_boxes(idx2)*0.5))*rand(1);
                this.y_boxes(idx2) = 0.001 + rand(1)*0.05;
                gap_min = this.d_boxes(idx1)/2 + this.d_boxes(idx2)/2 + this.toll_contatto;
                if abs(this.x_boxes(idx2)-this.x_boxes(idx1)) < gap_min
                    this.x_boxes(idx2) = this.x_boxes(idx1) + gap_min + 0.05;
                end
                this.x_boxes(idx2) = max(this.d_boxes(idx2)/2, ...
                    min(L-this.d_boxes(idx2)/2, this.x_boxes(idx2)));
                if abs(this.x_boxes(idx2)-this.x_boxes(idx1)) < gap_min
                    this.x_boxes(idx1) = this.x_boxes(idx2) - gap_min - 0.05;
                end
                this.x_boxes(idx1) = max(this.d_boxes(idx1)/2, ...
                    min(L-this.d_boxes(idx1)/2, this.x_boxes(idx1)));
                this.x_boxes_prec(idx1) = this.x_boxes(idx1);
                this.y_boxes_prec(idx1) = this.y_boxes(idx1);
                this.x_boxes_prec(idx2) = this.x_boxes(idx2);
                this.y_boxes_prec(idx2) = this.y_boxes(idx2);
            else
                idx = this.n_boxes_tot;
                this.x_boxes(idx) = this.d_boxes(idx)*0.55 + ...
                    (L-this.d_boxes(idx)*0.55)*rand(1);
                this.y_boxes(idx) = 0.001 + rand(1)*0.01;
                this.x_boxes_prec(idx) = this.x_boxes(idx);
                this.y_boxes_prec(idx) = this.y_boxes(idx);
            end
        end

        % ----- Collisioni laterali [FIX-2] --------------------------------
        function this = risolviColisioniLaterali(this)
            MAX_ITER = 5;
            for iter = 1:MAX_ITER
                trovata = false;
                for ii = 1:this.n_boxes_tot
                    if this.x_boxes(ii) < 0, continue; end
                    for jj = ii+1:this.n_boxes_tot
                        if this.x_boxes(jj) < 0, continue; end
                        dx     = this.x_boxes(ii) - this.x_boxes(jj);
                        min_dx = (this.d_boxes(ii) + this.d_boxes(jj)) / 2;
                        if abs(dx) < min_dx && abs(dx) > 1e-9
                            trovata = true;
                            nx  = sign(dx);
                            pen = min_dx - abs(dx) + this.toll_contatto;
                            this.x_boxes(ii) = this.x_boxes(ii) + nx*pen/2;
                            this.x_boxes(jj) = this.x_boxes(jj) - nx*pen/2;
                            dvx = (this.vx_boxes(ii) - this.vx_boxes(jj)) * nx;
                            if dvx < 0
                                j_imp = -(1+this.coeff_restituzione)*dvx/2;
                                this.vx_boxes(ii) = this.vx_boxes(ii) + j_imp*nx;
                                this.vx_boxes(jj) = this.vx_boxes(jj) - j_imp*nx;
                            end
                        end
                    end
                end
                if ~trovata, break; end
            end
        end

    end % private

    methods (Access = protected)
        function envUpdatedCallback(this) %#ok<MANU>
>>>>>>> Stashed changes
        end
    end
end
