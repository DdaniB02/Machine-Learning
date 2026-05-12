classdef RL_environment1 < rl.env.MATLABEnvironment
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
        % level 0→1 in ~120 episodi (step=0.008/ep)
        % Fase 1 [0.00-0.40):  2 pacchi, diam fisso   → ~50 ep
        % Fase 2 [0.40-0.80):  pacchi crescenti        → ~50 ep
        % Fase 3 [0.80-1.00]:  configurazione piena    → da ep ~100 in poi
        %
        % Motivazione: con step=0.002 la fase 1 durava 200 ep → l'agente
        % saturava il caso banale e si cristallizzava prima di vedere
        % scenari più complessi. Con step=0.008 il curriculum sale più
        % velocemente, forzando l'agente ad adattarsi continuamente.
        curriculum_level = 0.0;
        curriculum_step  = 0.008;

        % --- Collisioni ---
        toll_contatto      = 0.002;
        coeff_restituzione = 0.1;

        % --- Tempo ---
        dt   = 0.01;
        time = 0;
        cont = 0;
    end

    properties (Hidden)
        VisualizeAnimation = false
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
            end
            ActionInfo.LowerLimit = lim_low;
            ActionInfo.UpperLimit = lim_up;

            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
        end

        % -----------------------------------------------------------------
        function [Observation, Reward, IsDone, Info] = step(this, Action)
            Info = [];
            this.time = this.time + this.dt;
            this.cont = this.cont + 1;

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
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end

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
                else
                    this.index_AMS(ii) = 25;
                end
            end

            % Tracciamento uscite (il treadmill è già gestito dalla cinematica:
            % quando i_b > 4 il pacco riceve vy = v_treadmill nello step di fisica)
            y_exit_line = H;
            y_eval_line = H + this.reward_area_length;
            this.index_exit_prev = this.index_exit;
            for ii = 1:this.n_boxes_tot
                if this.x_boxes(ii) < 0, continue; end
                if this.y_boxes(ii) > y_exit_line && this.pack_exited(ii) == 0
                    this.pack_exited(ii) = 1;
                    this.index_exit      = this.index_exit + 1;
                    this.exit_order(this.index_exit) = ii;
                    this.in_reward_area(ii) = 1;
                end
                % Aggiorna flag area di reward (per il visualizzatore)
                if this.in_reward_area(ii) && ...
                        this.y_boxes(ii) - this.d_boxes(ii)/2 > y_eval_line
                    this.in_reward_area(ii) = 0;
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
            this.time = 0;
            this.cont = 0;
            this.next_gen_step = 40 + round(rand()*60);





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
                end
            end

            InitialObservation = buildObservation(this);
            this.State = InitialObservation;

            notifyEnvUpdated(this);
        end

    end % methods necessari

    %% =====================================================================
    %  Metodi pubblici opzionali
    %% =====================================================================
    methods

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
            if nargout, varargout{1} = this.Visualizer; end
            this.VisualizeAnimation = true;
            this.VisualizeActions   = false;
            this.VisualizeStates    = false;
        end

        % -----------------------------------------------------------------
        % Reward — tre componenti dense, tutte in [0,1] per step
        %
        % Analisi del problema precedente:
        %   r_X = tanh(|dx|/r_pair) era quasi sempre ~1 perché i pacchi
        %   partono già separati in X → nessun gradiente per la singolazione.
        %   r_sing si attivava raramente (pacchi nell'area post-AMS per <53 step).
        %   Risultato: l'agente imparava solo "vai dritto veloce" (r_Y).
        %
        % Nuova struttura:
        %
        % A) AVANZAMENTO IN Y  [peso 0.5]
        %    r_Y ∈ [0,1]: progresso medio in Y / (av_max * dt)
        %    Segnale denso e fondamentale — i pacchi devono attraversare la griglia.
        %
        % B) SINGOLAZIONE IN GRIGLIA  [peso 0.5]
        %    Per ogni coppia (i,j) in griglia, misura se i pacchi sono
        %    in "fila indiana": separati in Y di almeno la somma dei raggi,
        %    indipendentemente dalla separazione in X.
        %
        %    gap_Y = |y_i - y_j| - (d_i + d_j)/2   [gap Y tra bounding box]
        %    score_pair = tanh(gap_Y / d_ref)
        %      > 0: pacchi separati in Y (buono)
        %      < 0: pacchi sovrapposti in Y (male → penalità)
        %
        %    r_B ∈ (-1, 1]: media su tutte le coppie, riscalata in [0,1].
        %
        % r_total = 0.5*r_Y + 0.5*r_B   ∈ [0, 1] per step
        %
        % Con 1000 step/episodio: reward attesa massima ≈ 1000
        % -----------------------------------------------------------------
        function Reward = getReward(this)
            av_max = this.upperlim_v;

            % --- A) Avanzamento in Y ----------------------------------------
            r_Y       = 0;
            n_in_grid = 0;
            for ii = 1:this.n_boxes_tot
                if this.x_boxes(ii) < 0 || this.pack_exited(ii), continue; end
                dy        = this.y_boxes(ii) - this.y_boxes_prec(ii);
                r_Y       = r_Y + max(0, dy);
                n_in_grid = n_in_grid + 1;
            end
            if n_in_grid > 0
                r_Y = r_Y / (n_in_grid * av_max * this.dt);  % ∈ [0,1]
            end

            % --- B) Singolazione in Y tra coppie in griglia -----------------
            % Misura il gap in Y tra ogni coppia di pacchi non ancora usciti.
            % Positivo = ben separati in Y (fila indiana).
            % Negativo = sovrapposti in Y (collisione frontale potenziale).
            r_B    = 0;
            n_pair = 0;
            d_ref  = mean(this.d_boxes(this.d_boxes > 0));  % diam medio
            if isempty(d_ref) || d_ref == 0, d_ref = this.d_AMS; end

            for ii = 1:this.n_boxes_tot
                if this.x_boxes(ii) < 0 || this.pack_exited(ii), continue; end
                for jj = ii+1:this.n_boxes_tot
                    if this.x_boxes(jj) < 0 || this.pack_exited(jj), continue; end
                    % Gap Y tra i bounding box (negativo = overlap)
                    gap_Y = abs(this.y_boxes(ii) - this.y_boxes(jj)) ...
                            - (this.d_boxes(ii) + this.d_boxes(jj)) / 2;
                    % Normalizza su d_ref: tanh > 0 se separati, < 0 se overlap
                    r_B    = r_B + tanh(gap_Y / d_ref);
                    n_pair = n_pair + 1;
                end
            end
            if n_pair > 0
                % tanh ∈ (-1,1) → riscala in [0,1]
                r_B = (r_B / n_pair + 1) / 2;
            else
                r_B = 0.5;  % nessuna coppia: neutro
            end

            % --- Reward totale ----------------------------------------------
            Reward = 0.5 * r_Y + 0.5 * r_B;
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
        end
    end

end