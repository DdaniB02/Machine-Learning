classdef AMSVisualizer < rl.env.viz.AbstractFigureVisualizer

    methods
        function this = AMSVisualizer(env)
            this = this@rl.env.viz.AbstractFigureVisualizer(env);
        end
    end

    %% Protected methods
    methods (Access = protected)
        function f = buildFigure(this)
            f = figure( ...
                Toolbar="none", ...
                Visible="on", ...
                HandleVisibility="off", ...
                NumberTitle="off", ...
                Name="AMS",...
                CloseRequestFcn=@(~,~)delete(this));
            if ~strcmp(f.WindowStyle, "docked")
                f.Position = [200 100 800 500];
            end

            % Turn the menubar off here instead of during construction to
            % run the animation while running from live script
            f.MenuBar = "none";

            ha = gca(f);
            ha.XLimMode = "manual";
            ha.YLimMode = "manual";
            ha.ZLimMode = "manual";
            ha.DataAspectRatioMode = "manual";
            ha.PlotBoxAspectRatioMode = "manual";
            ha.XLim = [0 1];
            ha.YLim = [0 1];
            hold(ha,"on");
        end

        function updatePlot(this)

            env = this.Environment;

            action = env.LastAction;
            rot_AMS = zeros(5,5);
            vel_AMS = zeros(5,5);

            upperlim_v = 2.2;
            lowerlim_v = 0.5;

            for ii=1:5
                for jj=1:5
                    rot_AMS(ii,jj) = action(jj*2-1+(ii-1)*10);
                    v_norm = action(jj*2+(ii-1)*10)/(upperlim_v-lowerlim_v) - lowerlim_v;
                    vel_AMS(ii,jj) = 0.05 + v_norm*(0.15-0.05);
                end
            end

            f = this.Figure;
            ha = gca(f);

            d_AMS = env.d_AMS;

            for ii=1:env.max_gen_boxes
                d_boxes(ii) = env.d_boxes(ii);
                x_boxes(ii) = env.x_boxes(ii);
                y_boxes(ii) = env.y_boxes(ii);
            end

            h1plot = findobj(ha,'Tag','h1plot');
            h2plot = findobj(ha,'Tag','h2plot');
            h3plot = findobj(ha,'Tag','h3plot');
            h4plot = findobj(ha,'Tag','h4plot');
            h5plot = findobj(ha,'Tag','h5plot');
            h6plot = findobj(ha,'Tag','h6plot');
            v1plot = findobj(ha,'Tag','v1plot');
            v2plot = findobj(ha,'Tag','v2plot');
            v3plot = findobj(ha,'Tag','v3plot');
            v4plot = findobj(ha,'Tag','v4plot');
            v5plot = findobj(ha,'Tag','v5plot');
            v6plot = findobj(ha,'Tag','v6plot');
            pkg1plot = findobj(ha,'Tag','pkg1plot');
            pkg2plot = findobj(ha,'Tag','pkg2plot');
            pkg3plot = findobj(ha,'Tag','pkg3plot');
            pkg4plot = findobj(ha,'Tag','pkg4plot');
            pkg5plot = findobj(ha,'Tag','pkg5plot');
            pkg6plot = findobj(ha,'Tag','pkg6plot');
            pkg7plot = findobj(ha,'Tag','pkg7plot');
            pkg8plot = findobj(ha,'Tag','pkg8plot');
            pkg9plot = findobj(ha,'Tag','pkg9plot');
            pkg10plot = findobj(ha,'Tag','pkg10plot');

            if isempty(h1plot) || ~isvalid(h1plot) || ...
                    isempty(h2plot) || ~isvalid(h2plot) || ...
                    isempty(h3plot) || ~isvalid(h3plot) || ...
                    isempty(h4plot) || ~isvalid(h4plot) || ...
                    isempty(h5plot) || ~isvalid(h5plot) || ...
                    isempty(h6plot) || ~isvalid(h6plot) || ...
                    isempty(v1plot) || ~isvalid(v1plot) || ...
                    isempty(v2plot) || ~isvalid(v2plot) || ...
                    isempty(v3plot) || ~isvalid(v3plot) || ...
                    isempty(v4plot) || ~isvalid(v4plot) || ...
                    isempty(v5plot) || ~isvalid(v5plot) || ...
                    isempty(v6plot) || ~isvalid(v6plot) || ...
                    isempty(pkg1plot) || ~isvalid(pkg1plot) || ...
                    isempty(pkg2plot) || ~isvalid(pkg2plot) || ...
                    isempty(pkg3plot) || ~isvalid(pkg3plot) || ...
                    isempty(pkg4plot) || ~isvalid(pkg4plot) || ...
                    isempty(pkg5plot) || ~isvalid(pkg5plot) || ...
                    isempty(pkg6plot) || ~isvalid(pkg6plot) || ...
                    isempty(pkg7plot) || ~isvalid(pkg7plot) || ...
                    isempty(pkg8plot) || ~isvalid(pkg8plot) || ...
                    isempty(pkg9plot) || ~isvalid(pkg9plot) || ...
                    isempty(pkg10plot) || ~isvalid(pkg10plot)

                h1plot = line(ha,ha.XLim,[0 0],'LineWidth',2,'Color','k','Tag','h1plot');
                h2plot = line(ha,ha.XLim,[d_AMS d_AMS],'LineWidth',2,'Color','k','Tag','h2plot');
                h3plot = line(ha,ha.XLim,[d_AMS*2 d_AMS*2],'LineWidth',2,'Color','k','Tag','h3plot');
                h4plot = line(ha,ha.XLim,[d_AMS*3 d_AMS*3],'LineWidth',2,'Color','k','Tag','h4plot');
                h5plot = line(ha,ha.XLim,[d_AMS*4 d_AMS*4],'LineWidth',2,'Color','k','Tag','h5plot');
                h6plot = line(ha,ha.XLim,[d_AMS*5 d_AMS*5],'LineWidth',2,'Color','k','Tag','h6plot');

                v1plot = line(ha,[0 0],ha.YLim,'LineWidth',2,'Color','k','Tag','v1plot');
                v2plot = line(ha,[d_AMS d_AMS],ha.YLim,'LineWidth',2,'Color','k','Tag','v2plot');
                v3plot = line(ha,[d_AMS*2 d_AMS*2],ha.YLim,'LineWidth',2,'Color','k','Tag','v3plot');
                v4plot = line(ha,[d_AMS*3 d_AMS*3],ha.YLim,'LineWidth',2,'Color','k','Tag','v4plot');
                v5plot = line(ha,[d_AMS*4 d_AMS*4],ha.YLim,'LineWidth',2,'Color','k','Tag','v5plot');
                v6plot = line(ha,[d_AMS*5 d_AMS*5],ha.YLim,'LineWidth',2,'Color','k','Tag','v6plot');

                pkg1plot = rectangle(ha,'Position',[x_boxes(1), y_boxes(1), d_boxes(1)/2, d_boxes(1)/2], 'FaceColor','g','Tag','pkg1plot');
                pkg2plot = rectangle(ha,'Position',[x_boxes(2), y_boxes(2), d_boxes(2)/2, d_boxes(2)/2], 'FaceColor','b','Tag','pkg2plot');
                pkg3plot = rectangle(ha,'Position',[x_boxes(3), y_boxes(3), d_boxes(3)/2, d_boxes(3)/2], 'FaceColor','r','Tag','pkg3plot');
                pkg4plot = rectangle(ha,'Position',[x_boxes(4), y_boxes(4), d_boxes(4)/2, d_boxes(4)/2], 'FaceColor','k','Tag','pkg4plot');
                pkg5plot = rectangle(ha,'Position',[x_boxes(5), y_boxes(5), d_boxes(5)/2, d_boxes(5)/2], 'FaceColor','c','Tag','pkg5plot');
                pkg6plot = rectangle(ha,'Position',[x_boxes(6), y_boxes(6), d_boxes(6)/2, d_boxes(6)/2], 'FaceColor','y','Tag','pkg6plot');
                pkg7plot = rectangle(ha,'Position',[x_boxes(7), y_boxes(7), d_boxes(7)/2, d_boxes(7)/2], 'FaceColor','m','Tag','pkg7plot');
                pkg8plot = rectangle(ha,'Position',[x_boxes(8), y_boxes(8), d_boxes(8)/2, d_boxes(8)/2], 'FaceColor','w','Tag','pkg8plot');
                pkg9plot = rectangle(ha,'Position',[x_boxes(9), y_boxes(9), d_boxes(9)/2, d_boxes(9)/2], 'FaceColor','g','Tag','pkg9plot');
                pkg10plot = rectangle(ha,'Position',[x_boxes(10), y_boxes(10), d_boxes(10)/2, d_boxes(10)/2], 'FaceColor','r','Tag','pkg10plot');

            end

            pkg1plot.Position = [x_boxes(1), y_boxes(1), d_boxes(1)/2, d_boxes(1)/2];
            pkg2plot.Position = [x_boxes(2), y_boxes(2), d_boxes(2)/2, d_boxes(2)/2];
            pkg3plot.Position = [x_boxes(3), y_boxes(3), d_boxes(3)/2, d_boxes(3)/2];
            pkg4plot.Position = [x_boxes(4), y_boxes(4), d_boxes(4)/2, d_boxes(4)/2];
            pkg5plot.Position = [x_boxes(5), y_boxes(5), d_boxes(5)/2, d_boxes(5)/2];
            pkg6plot.Position = [x_boxes(6), y_boxes(6), d_boxes(6)/2, d_boxes(6)/2];
            pkg7plot.Position = [x_boxes(7), y_boxes(7), d_boxes(7)/2, d_boxes(7)/2];
            pkg8plot.Position = [x_boxes(8), y_boxes(8), d_boxes(8)/2, d_boxes(8)/2];
            pkg9plot.Position = [x_boxes(9), y_boxes(9), d_boxes(9)/2, d_boxes(9)/2];
            pkg10plot.Position = [x_boxes(10), y_boxes(10), d_boxes(10)/2, d_boxes(10)/2];

            % Refresh rendering in figure window
            drawnow();
        end
    end
end