classdef RL_environment1 < rl.env.MATLABEnvironment
    %AMS_RL: Ambiente RL per smistamento automatico di pacchi su matrice AMS.
    %
    % MIGLIORAMENTI rispetto alla versione originale:
    %   FISICA:
    %     - Collisioni rilevate PRIMA del movimento (swept collision detection)
    %       per evitare il tunnel effect a velocità elevate
    %     - Risoluzione collisioni lungo la normale al contatto (MTV -
    %       Minimum Translation Vector), non più solo su X o Y separati
    %     - Loop collisioni separato dalla cinematica (due passate distinte)
    %     - Boundary clamp applicato DOPO la risoluzione collisioni
    %     - Rimossa x_boxes_vect / y_boxes_vect (crescita illimitata in memoria);
    %       sostituita con buffer circolare a 2 slot (t e t-1)
    %
    %   REWARD:
    %     - Premio uscita pacco: delta (solo allo step dell'evento), non cumulativo
    %     - Premio singolazione: bonus quando due pacchi superano la soglia
    %       minima di separazione laterale all'uscita
    %     - Premio throughput: reward proporzionale alla velocità media Y
    %       dei pacchi ancora in griglia
    %     - Penalità collisione proporzionale alla profondità di penetrazione
    %     - Penalità stazionamento invariata (shaping costante)

    %% Properties
    properties
        % --- Parametri fisici AMS ---
        v_treadmill  = 0.6;          % m/s  velocità tapis roulant (oltre griglia)
        d_AMS        = 0.2;          % m    lato di ogni cella AMS
        n_i_AMS      = 5;            % celle lungo Y
        n_j_AMS      = 5;            % celle lungo X
        n_actions_art = 2;           % azioni per cella (rot, vel)

        upperlim_rot = pi*45/180;    % rad  limite sup. rotazione
        upperlim_v   = 2.2;          % m/s  limite sup. velocità
        lowerlim_rot = -pi*45/180;   % rad  limite inf. rotazione
        lowerlim_v   = 0.5;          % m/s  limite inf. velocità

        % --- Stato cinematico pacchi ---
        x_boxes      = zeros(10,1);  % posizione X corrente
        y_boxes      = zeros(10,1);  % posizione Y corrente
        x_boxes_prec = zeros(10,1);  % posizione X al passo precedente
        y_boxes_prec = zeros(10,1);  % posizione Y al passo precedente
        vx_boxes     = zeros(10,1);  % velocità X corrente  (m/s)
        vy_boxes     = zeros(10,1);  % velocità Y corrente  (m/s)
        d_boxes      = zeros(10,1);  % diametro pacco (trattato come quadrato)

        % --- Indici AMS correnti per ogni pacco ---
        i_box = zeros(10,1);         % riga  cella AMS (1-5, 6 = fuori griglia)
        j_box = zeros(10,1);         % colonna cella AMS
        index_AMS = zeros(10,1);

        % --- Ultima azione de-normalizzata ---
        LastAction = zeros(50,1);

        % --- Gestione pacchi ---
        n_boxes_tot     = 0;         % pacchi attualmente attivi
        max_gen_boxes   = 10;        % max pacchi simultanei
        cont_new_box    = 0;         % contatore eventi di generazione
        n_boxes_tot_vect = [];

        % --- Uscite ---
        pack_exited  = zeros(10,1);  % flag: pacco ii è uscito?
        exit_order   = zeros(10,1);  % ordine di uscita
        index_exit   = 0;            % quanti pacchi sono usciti finora
        index_exit_prev = 0;         % valore al passo precedente (per delta reward)
        y_exit       = zeros(10,1);  % posizione Y congelata al momento dell'uscita

        % --- Curriculum learning ---
        % La difficoltà aumenta progressivamente: all'inizio i diametri sono
        % uniformi e le posizioni meno casuali, poi aumenta la varianza.
        % curriculum_level: 0.0 (facile) → 1.0 (difficoltà piena)
        curriculum_level = 0.0;      % aggiornato da RL_main dopo ogni episodio
        curriculum_step  = 0.002;    % incremento per episodio

        % --- Parametri collisione ---
        toll_contatto      = 0.002;  % m  gap minimo di separazione post-risoluzione
        coeff_restituzione = 0.1;    % 0=anelastico (cartone), 1=elastico

        % ---------------------------------------------------------------
        % PESI REWARD — modifica qui per cambiare il comportamento
        % ---------------------------------------------------------------
        % 1. Penalità collisione base (moltiplicata per la penetrazione)
        %    Valore negativo. Es: -50 → penalità lieve, -200 → molto severa.
        rw_collisione_base   = -50;

        % 2. Premio uscita pacco (dato una sola volta per pacco)
        %    Valore positivo. Es: 150 → obiettivo principale dell'agente.
        rw_uscita            = 150;

        % 3a. Premio singolazione: pacchi ben separati verticalmente
        %     (gap Y > somma diametri)
        rw_singolazione_ok   = 80;

        % 3b. Premio singolazione parziale (gap Y > metà somma diametri)
        rw_singolazione_parz = 20;

        % 4. Shaping progressivo per riga AMS (moltiplicato per la riga 1-5)
        %    Es: 0.5 → riga 1 dà +0.5/step, riga 5 dà +2.5/step
        rw_shaping_riga      = 0.5;

        % 5. Premio throughput (moltiplicato per velocità media Y normalizzata)
        %    Es: 8 → incentiva avanzamento rapido verso l'uscita
        rw_throughput        = 8;

        % 6. Penalità stazionamento (costo fisso per ogni step)
        %    Valore negativo. Es: -0.5 → piccola spinta a non restare fermi
        rw_stazionamento     = -0.5;

        % 7. Premio allineamento in fila indiana (dato ogni step per coppia)
        %    Condizioni simultanee richieste:
        %      a) pacchi separati verticalmente: |dy| > r_i + r_j  (non si toccano)
        %      b) pacchi allineati lateralmente: |dx| < soglia_x    (stessa corsia)
        %      c) entrambi ancora in griglia (non usciti)
        %    Tenere piccolo rispetto a rw_uscita per evitare reward hacking
        %    (agente che tiene i pacchi fermi e allineati senza farli avanzare).
        %    Es: 10 → per 2 pacchi allineati per 100 step = +2000,
        %        contro rw_uscita=150 × 10 pacchi = +1500 → attenzione al bilanciamento
        rw_allineamento      = 4;

        % Soglia laterale per considerare due pacchi "sulla stessa corsia".
        % Espressa come multiplo della somma dei raggi: 1.0 = devono stare
        % entro un diametro medio di distanza X, 2.0 = più permissivo.
        rw_allineamento_soglia_x = 1.5;

        % --- Tempo ---
        dt   = 0.01;                 % s  timestep
        time = 0;                    % s  tempo simulazione
        cont = 0;                    % contatore step
    end

    properties (Hidden)
        VisualizeAnimation = false
        VisualizeActions   = false
        VisualizeStates    = false
    end

    properties
        % Vettore di stato RL (50 elementi)
        State = zeros(50,1)
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

        % -----------------------------------------------------------------
        % Costruttore
        % -----------------------------------------------------------------
        function this = RL_environment1()
            numObservations = 50;
            ObservationInfo = rlNumericSpec([numObservations 1]);
            ObservationInfo.Name        = 'Parcel States';
            ObservationInfo.Description = 'x, y, x_prec, y_prec, diametro per ciascuno dei 10 pacchi';

            n_actions  = 5*5*2;   % 50
            ActionInfo = rlNumericSpec([n_actions 1]);
            ActionInfo.Name        = 'AMS Action';
            ActionInfo.Description = 'r1,v1, r2,v2, ... per ogni cella della matrice 5x5';

            lim_low = zeros(n_actions,1);
            lim_up  = zeros(n_actions,1);
            for ii = 1:2:n_actions
                lim_up(ii)    =  pi*45/180;   % rot max
                lim_up(ii+1)  =  2.2;         % vel max
                lim_low(ii)   = -pi*45/180;   % rot min
                lim_low(ii+1) =  0.5;         % vel min
            end
            ActionInfo.LowerLimit = lim_low;
            ActionInfo.UpperLimit = lim_up;

            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
        end

        % -----------------------------------------------------------------
        % Step
        % -----------------------------------------------------------------
        function [Observation, Reward, IsDone, Info] = step(this, Action)
            Info = [];

            this.time = this.time + this.dt;
            this.cont = this.cont + 1;

            L = this.d_AMS * this.n_j_AMS;   % larghezza totale griglia [m]

            % --- De-normalizzazione azioni [-1,1] → limiti fisici ----------
            lim_up  = repmat([this.upperlim_rot; this.upperlim_v], 25, 1);
            lim_low = repmat([this.lowerlim_rot; this.lowerlim_v], 25, 1);
            AMS_actions = lim_low + (1 + Action) .* (lim_up - lim_low) ./ 2;
            AMS_actions = max(lim_low, min(lim_up, AMS_actions));
            this.LastAction = AMS_actions;

            % Estrai matrici rotazione e velocità 5×5
            rotation_AMS = zeros(5,5);
            v_AMS        = zeros(5,5);
            for ii = 1:this.n_i_AMS
                for jj = 1:this.n_j_AMS
                    rotation_AMS(ii,jj) = AMS_actions(jj*2-1 + (ii-1)*10);
                    v_AMS(ii,jj)        = AMS_actions(jj*2   + (ii-1)*10);
                end
            end

            % --- Generazione nuovi pacchi ogni 0.75 s (max 2 per evento) --
            if mod(round(this.time,2), 0.75) == 0 && ...
                    this.n_boxes_tot < this.max_gen_boxes && this.time > 0.25
                this = generaPacki(this, L);
            end

            % Pacchi non ancora generati → coordinate sentinella
            for ii = this.n_boxes_tot+1 : this.max_gen_boxes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end

            % --- Salva posizioni precedenti --------------------------------
            this.x_boxes_prec(1:this.n_boxes_tot) = this.x_boxes(1:this.n_boxes_tot);
            this.y_boxes_prec(1:this.n_boxes_tot) = this.y_boxes(1:this.n_boxes_tot);

            % --- PASSATA 1: Cinematica (aggiorna posizioni) ----------------
            for ii = 1:this.n_boxes_tot
                [i_b, j_b] = getAMSIndex(this, this.x_boxes(ii), this.y_boxes(ii));
                this.i_box(ii) = i_b;
                this.j_box(ii) = j_b;

                if i_b < 6
                    vx = v_AMS(i_b,j_b) * sin(rotation_AMS(i_b,j_b));
                    vy = v_AMS(i_b,j_b) * cos(rotation_AMS(i_b,j_b));
                else
                    % Pacco fuori griglia: tapis roulant solo in Y
                    vx = 0;
                    vy = this.v_treadmill;
                end

                this.vx_boxes(ii) = vx;
                this.vy_boxes(ii) = vy;

                this.x_boxes(ii) = this.x_boxes(ii) + vx * this.dt;
                this.y_boxes(ii) = this.y_boxes(ii) + vy * this.dt;
            end

            % --- PASSATA 2: Risoluzione collisioni (MTV + restituzione) ----
            this = risolviCollisioni(this);

            % --- Clamp bordi laterali DOPO la risoluzione collisioni -------
            for ii = 1:this.n_boxes_tot
                r = this.d_boxes(ii) / 2;
                this.x_boxes(ii) = max(r, min(L - r, this.x_boxes(ii)));
            end

            % --- Aggiorna indici AMS post-movimento -----------------------
            for ii = 1:this.n_boxes_tot
                [i_b, j_b] = getAMSIndex(this, this.x_boxes(ii), this.y_boxes(ii));
                this.i_box(ii) = i_b;
                this.j_box(ii) = j_b;
                if i_b < 6 && j_b < 6
                    this.index_AMS(ii) = j_b + (i_b-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end
            end

            % --- Costruzione osservazione ----------------------------------
            Observation = buildObservation(this);
            this.State  = Observation;

            % --- Verifica uscite -------------------------------------------
            this.index_exit_prev = this.index_exit;
            for ii = 1:this.n_boxes_tot
                if this.y_boxes(ii) > this.d_AMS*5 && this.pack_exited(ii) == 0
                    this.pack_exited(ii) = 1;
                    this.index_exit = this.index_exit + 1;
                    this.exit_order(this.index_exit) = ii;
                    % Congela la Y al momento dell'uscita: usata nella reward
                    % di singolazione per evitare che sep_y cresca ad ogni step
                    this.y_exit(ii) = this.y_boxes(ii);
                end
            end

            % --- Condizione terminale: tutti i pacchi sono usciti ---------
            IsDone = (sum(this.pack_exited(1:this.n_boxes_tot)) == this.max_gen_boxes);
            this.IsDone = IsDone;

            % --- Reward ---------------------------------------------------
            Reward = getReward(this, AMS_actions);

            notifyEnvUpdated(this);
        end

        % -----------------------------------------------------------------
        % Reset
        % -----------------------------------------------------------------
        function InitialObservation = reset(this)
            this.index_exit      = 0;
            this.index_exit_prev = 0;
            this.pack_exited     = zeros(10,1);
            this.exit_order      = zeros(10,1);
            this.y_exit          = zeros(10,1);

            % Aggiorna curriculum: ogni reset la difficoltà sale fino a 1.0
            this.curriculum_level = min(1.0, this.curriculum_level + this.curriculum_step);
            this.cont_new_box    = 1;
            this.n_boxes_tot_vect = 2;
            this.n_boxes_tot     = 2;
            this.time = 0;
            this.cont = 0;

            L         = this.d_AMS * this.n_j_AMS;
            max_d_box = 2.0  * this.d_AMS;
            min_d_box = 0.25 * this.d_AMS;

            % Curriculum: a livello basso tutti i pacchi hanno diametro medio
            % (bassa varianza), a livello pieno varianza completa.
            d_medio = (max_d_box + min_d_box) / 2;
            for ii = 1:this.max_gen_boxes
                d_rand = min_d_box + (max_d_box - min_d_box)*rand(1);
                % Interpola tra diametro fisso e casuale in base al curriculum
                this.d_boxes(ii) = (1 - this.curriculum_level)*d_medio + ...
                                        this.curriculum_level * d_rand;
            end

            % Posiziona i 2 pacchi iniziali (metà sinistra e metà destra)
            x1 = this.d_boxes(1)*0.5 + (this.d_AMS*2.5 - this.d_boxes(1)*0.5)*rand(1);
            x2 = this.d_AMS*2.5 + this.d_boxes(2)*0.5 + ...
                 (L - (this.d_AMS*2.5 + this.d_boxes(2)*0.5))*rand(1);

            % Risolvi eventuale sovrapposizione iniziale
            gap_min = this.d_boxes(1)/2 + this.d_boxes(2)/2 + this.toll_contatto;
            if abs(x2-x1) < gap_min
                x2 = x1 + gap_min + 0.05;
            end
            x2 = max(this.d_boxes(2)/2, min(L - this.d_boxes(2)/2, x2));
            if abs(x2-x1) < gap_min
                x1 = x2 - gap_min - 0.05;
            end

            y1 = 0.001 + rand(1)*0.01;
            y2 = 0.001 + rand(1)*0.05;

            this.x_boxes(1) = x1;  this.y_boxes(1) = y1;
            this.x_boxes(2) = x2;  this.y_boxes(2) = y2;

            this.x_boxes_prec(1) = x1; this.y_boxes_prec(1) = y1;
            this.x_boxes_prec(2) = x2; this.y_boxes_prec(2) = y2;

            this.vx_boxes = zeros(10,1);
            this.vy_boxes = zeros(10,1);

            % Pacchi non ancora generati → sentinella
            for ii = 3:this.max_gen_boxes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
                this.x_boxes_prec(ii) = -1;
                this.y_boxes_prec(ii) = -1;
            end

            % Calcola indici AMS iniziali
            this.index_AMS = zeros(10,1);
            for ii = 1:this.n_boxes_tot
                [i_b, j_b] = getAMSIndex(this, this.x_boxes(ii), this.y_boxes(ii));
                this.i_box(ii) = i_b;
                this.j_box(ii) = j_b;
                if i_b < 6 && j_b < 6
                    this.index_AMS(ii) = j_b + (i_b-1)*5;
                end
            end

            InitialObservation = buildObservation(this);
            this.State = InitialObservation;

            notifyEnvUpdated(this);
        end

    end % methods (necessari)

    %% =====================================================================
    %  Metodi opzionali pubblici
    %% =====================================================================
    methods

        function varargout = plot(this)
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = AMSVisualizer(this);
            else
                bringToFront(this.Visualizer);
            end
            if nargout, varargout{1} = this.Visualizer; end
            this.VisualizeAnimation = true;
            this.VisualizeActions   = false;
            this.VisualizeStates    = false;
        end

        % -----------------------------------------------------------------
        % Reward function
        % Per modificare i pesi, cambia le proprietà rw_* nella sezione
        % "PESI REWARD" in cima al file — non serve toccare questo metodo.
        % -----------------------------------------------------------------
        function Reward = getReward(this, ~)
            Reward = 0;

            % 1. PENALITÀ COLLISIONE proporzionale alla penetrazione
            for ii = 1:this.n_boxes_tot
                for jj = ii+1 : this.n_boxes_tot
                    if this.x_boxes(ii) < 0 || this.x_boxes(jj) < 0, continue; end
                    dx = this.x_boxes(ii) - this.x_boxes(jj);
                    dy = this.y_boxes(ii) - this.y_boxes(jj);
                    dist     = sqrt(dx^2 + dy^2);
                    min_dist = (this.d_boxes(ii) + this.d_boxes(jj)) / 2;
                    if dist < min_dist
                        penetrazione = min_dist - dist;
                        Reward = Reward + this.rw_collisione_base * (1 + 3*(penetrazione/min_dist));
                    end
                end
            end

            % 2. PREMIO USCITA (delta: solo allo step in cui il pacco esce)
            n_nuovi_usciti = this.index_exit - this.index_exit_prev;
            Reward = Reward + n_nuovi_usciti * this.rw_uscita;

            % 3. PREMIO SINGOLAZIONE verticale (usa y_exit congelata)
            if this.index_exit >= 2
                for kk = max(1, this.index_exit_prev+1) : this.index_exit
                    idx_uscito = this.exit_order(kk);
                    for mm = 1:kk-1
                        idx_prec = this.exit_order(mm);
                        if idx_uscito > 0 && idx_prec > 0
                            sep_y  = abs(this.y_exit(idx_uscito) - this.y_exit(idx_prec));
                            soglia = (this.d_boxes(idx_uscito) + this.d_boxes(idx_prec));
                            if sep_y > soglia
                                Reward = Reward + this.rw_singolazione_ok;
                            elseif sep_y > soglia * 0.5
                                Reward = Reward + this.rw_singolazione_parz;
                            end
                        end
                    end
                end
            end

            % 4. SHAPING PROGRESSIVO per riga AMS (gradiente denso)
            for ii = 1:this.n_boxes_tot
                if this.pack_exited(ii) == 0 && this.x_boxes(ii) >= 0
                    Reward = Reward + this.i_box(ii) * this.rw_shaping_riga;
                end
            end

            % 5. PREMIO THROUGHPUT (velocità media Y dei pacchi in griglia)
            n_in_griglia = 0;
            progresso_y  = 0;
            for ii = 1:this.n_boxes_tot
                if this.pack_exited(ii) == 0 && this.x_boxes(ii) >= 0
                    dy_step = this.y_boxes(ii) - this.y_boxes_prec(ii);
                    progresso_y  = progresso_y + max(0, dy_step);
                    n_in_griglia = n_in_griglia + 1;
                end
            end
            if n_in_griglia > 0
                Reward = Reward + this.rw_throughput * (progresso_y / n_in_griglia) / this.dt;
            end

            % 6. PREMIO ALLINEAMENTO IN FILA INDIANA
            %    Premia ogni coppia di pacchi che è contemporaneamente:
            %      a) separata verticalmente (gap Y > somma raggi → no collisione)
            %      b) allineata lateralmente  (gap X < soglia → stessa corsia)
            %      c) entrambi ancora in griglia
            %    Il premio è dato ad ogni step: incentiva il mantenimento
            %    della configurazione a fila indiana durante tutta la traversata.
            for ii = 1:this.n_boxes_tot
                for jj = ii+1 : this.n_boxes_tot
                    % Salta pacchi non ancora generati o già usciti
                    if this.x_boxes(ii) < 0 || this.x_boxes(jj) < 0, continue; end
                    if this.pack_exited(ii) || this.pack_exited(jj),  continue; end

                    dx = abs(this.x_boxes(ii) - this.x_boxes(jj));
                    dy = abs(this.y_boxes(ii) - this.y_boxes(jj));

                    somma_raggi = (this.d_boxes(ii) + this.d_boxes(jj)) / 2;
                    soglia_x    = somma_raggi * this.rw_allineamento_soglia_x;

                    sep_v_ok  = dy > somma_raggi;   % a) separati verticalmente
                    allin_ok  = dx < soglia_x;       % b) allineati lateralmente

                    if sep_v_ok && allin_ok
                        Reward = Reward + this.rw_allineamento;
                    end
                end
            end

            % 7. PENALITÀ STAZIONAMENTO (costo fisso per step)
            Reward = Reward + this.rw_stazionamento;
        end

        % -----------------------------------------------------------------
        % Set State (validazione)
        % -----------------------------------------------------------------
        function set.State(this, state)
            validateattributes(state, {'numeric'}, ...
                {'finite','real','vector','numel',50}, '', 'State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end

    end % methods (opzionali pubblici)

    %% =====================================================================
    %  Metodi privati (helper)
    %% =====================================================================
    methods (Access = private)

        % -----------------------------------------------------------------
        % Costruisce il vettore osservazione normalizzato [0,1]
        % -----------------------------------------------------------------
        function obs = buildObservation(this)
            L = this.d_AMS * this.n_j_AMS;
            H = this.d_AMS * this.n_i_AMS;
            obs = [ this.x_boxes / L;           ... % 1-10:  X norm
                    this.y_boxes / H;           ... % 11-20: Y norm
                    this.x_boxes_prec / L;      ... % 21-30: X_prec norm
                    this.y_boxes_prec / H;      ... % 31-40: Y_prec norm
                    this.d_boxes / (2*this.d_AMS) ]; % 41-50: diam norm
        end

        % -----------------------------------------------------------------
        % Restituisce la cella AMS (i,j) dato (x,y).
        % i=6 o j=6 significa fuori dalla griglia.
        % -----------------------------------------------------------------
        function [i_b, j_b] = getAMSIndex(this, x, y)
            d = this.d_AMS;
            if     y <= d,   i_b = 1;
            elseif y <= 2*d, i_b = 2;
            elseif y <= 3*d, i_b = 3;
            elseif y <= 4*d, i_b = 4;
            elseif y <= 5*d, i_b = 5;
            else,            i_b = 6;
            end

            if i_b == 6
                j_b = 6;
                return
            end

            if     x <= d,   j_b = 1;
            elseif x <= 2*d, j_b = 2;
            elseif x <= 3*d, j_b = 3;
            elseif x <= 4*d, j_b = 4;
            else,            j_b = 5;
            end
        end

        % -----------------------------------------------------------------
        % Generazione pacchi durante la simulazione
        % -----------------------------------------------------------------
        function this = generaPacki(this, L)
            this.cont_new_box = this.cont_new_box + 1;

            gen_n = min(2, this.max_gen_boxes - this.n_boxes_tot);
            if gen_n <= 0, return; end

            this.n_boxes_tot = this.n_boxes_tot + gen_n;
            this.n_boxes_tot_vect(this.cont_new_box) = gen_n;

            if gen_n >= 2
                % Pacco sinistro
                idx1 = this.n_boxes_tot - 1;
                this.x_boxes(idx1) = this.d_boxes(idx1)*0.5 + ...
                    (this.d_AMS*2.5 - this.d_boxes(idx1)*0.5)*rand(1);
                this.y_boxes(idx1) = 0.001 + rand(1)*0.01;

                % Pacco destro
                idx2 = this.n_boxes_tot;
                this.x_boxes(idx2) = this.d_AMS*2.5 + this.d_boxes(idx2)*0.5 + ...
                    (L - (this.d_AMS*2.5 + this.d_boxes(idx2)*0.5))*rand(1);
                this.y_boxes(idx2) = 0.001 + rand(1)*0.05;

                % Garantisci separazione minima tra i due nuovi pacchi
                gap_min = this.d_boxes(idx1)/2 + this.d_boxes(idx2)/2 + this.toll_contatto;
                if abs(this.x_boxes(idx2) - this.x_boxes(idx1)) < gap_min
                    this.x_boxes(idx2) = this.x_boxes(idx1) + gap_min + 0.05;
                end
                this.x_boxes(idx2) = max(this.d_boxes(idx2)/2, ...
                    min(L - this.d_boxes(idx2)/2, this.x_boxes(idx2)));
                if abs(this.x_boxes(idx2) - this.x_boxes(idx1)) < gap_min
                    this.x_boxes(idx1) = this.x_boxes(idx2) - gap_min - 0.05;
                end

                this.x_boxes_prec(idx1) = this.x_boxes(idx1);
                this.y_boxes_prec(idx1) = this.y_boxes(idx1);
                this.x_boxes_prec(idx2) = this.x_boxes(idx2);
                this.y_boxes_prec(idx2) = this.y_boxes(idx2);
            else
                idx = this.n_boxes_tot;
                this.x_boxes(idx) = this.d_boxes(idx)*0.55 + ...
                    (L - this.d_boxes(idx)*0.55)*rand(1);
                this.y_boxes(idx) = 0.001 + rand(1)*0.01;
                this.x_boxes_prec(idx) = this.x_boxes(idx);
                this.y_boxes_prec(idx) = this.y_boxes(idx);
            end
        end

        % -----------------------------------------------------------------
        % Risoluzione collisioni con MTV (Minimum Translation Vector)
        %
        % Tratta ogni pacco come un disco di diametro d_boxes(ii).
        % Per ogni coppia (ii,jj) in collisione:
        %   1. Calcola il vettore normale n = (xi-xj)/dist
        %   2. Sposta i due centri lungo n di penetrazione/2 ciascuno
        %   3. Applica impulso di velocità lungo n (rispettando
        %      il coefficiente di restituzione)
        %
        % Il ciclo viene ripetuto fino a MAX_ITER volte per gestire
        % collisioni multiple simultanee (catena di pacchi).
        % -----------------------------------------------------------------
        function this = risolviCollisioni(this)
            MAX_ITER = 5;   % iterazioni massime di risoluzione per step

            for iter = 1:MAX_ITER
                collisione_trovata = false;

                for ii = 1:this.n_boxes_tot
                    if this.x_boxes(ii) < 0, continue; end   % sentinella

                    for jj = ii+1 : this.n_boxes_tot
                        if this.x_boxes(jj) < 0, continue; end

                        dx = this.x_boxes(ii) - this.x_boxes(jj);
                        dy = this.y_boxes(ii) - this.y_boxes(jj);
                        dist = sqrt(dx^2 + dy^2);
                        min_dist = (this.d_boxes(ii) + this.d_boxes(jj))/2;

                        if dist < min_dist && dist > 1e-9
                            collisione_trovata = true;

                            % --- Normale al contatto (da jj verso ii) ---
                            nx = dx / dist;
                            ny = dy / dist;

                            % --- Correzione posizione (MTV simmetrico) ---
                            penetrazione = min_dist - dist + this.toll_contatto;
                            this.x_boxes(ii) = this.x_boxes(ii) + nx * penetrazione/2;
                            this.y_boxes(ii) = this.y_boxes(ii) + ny * penetrazione/2;
                            this.x_boxes(jj) = this.x_boxes(jj) - nx * penetrazione/2;
                            this.y_boxes(jj) = this.y_boxes(jj) - ny * penetrazione/2;

                            % --- Correzione velocità (impulso 1D lungo n) ---
                            % Velocità relative lungo la normale
                            dvx = this.vx_boxes(ii) - this.vx_boxes(jj);
                            dvy = this.vy_boxes(ii) - this.vy_boxes(jj);
                            vrel_n = dvx*nx + dvy*ny;

                            % Applica impulso solo se i pacchi si avvicinano
                            if vrel_n < 0
                                % Massa unitaria → impulso simmetrico
                                j_imp = -(1 + this.coeff_restituzione) * vrel_n / 2;
                                this.vx_boxes(ii) = this.vx_boxes(ii) + j_imp*nx;
                                this.vy_boxes(ii) = this.vy_boxes(ii) + j_imp*ny;
                                this.vx_boxes(jj) = this.vx_boxes(jj) - j_imp*nx;
                                this.vy_boxes(jj) = this.vy_boxes(jj) - j_imp*ny;
                            end
                        end
                    end
                end

                if ~collisione_trovata, break; end
            end
        end

    end % methods (Access = private)

    %% =====================================================================
    %  Callback visualizzazione (opzionale)
    %% =====================================================================
    methods (Access = protected)
        function envUpdatedCallback(this) %#ok<MANU>
            % Lasciato intenzionalmente vuoto:
            % la diagnostica disp() ad ogni step rallenta enormemente il
            % training. Decommentare solo per debug locale.
            % disp(['Step: ', num2str(this.cont), ...
            %        '  Pacchi: ', num2str(this.n_boxes_tot), ...
            %        '  Usciti: ', num2str(this.index_exit)]);
        end
    end

end