%uses Backpropagation to find the parameter-values of W_1, W_2, B_1, B_2 of
%a three-layer-MLP with tanh(x) activation-function for (Nu|N|Ny)- MLP;
%editional it could be extended for M meassurements.
%P. Schmälzle 20.05.2017
%
%rigth now only for M = 1

clc;
clear;
%choose your input method
source = input('Input source = ?: File = 0, Keyboard = 1: ');
fprintf('source: %d\n',source);

if ~source
    filename = input('please enter the filename .mat: ','s');
end


%% initialization 
%Levenberg-Marquardt active
LM = input('Do you like to aktivate Levenberg-Marquardt? Yes = 1, No = 0 ');
fprintf('Method LM: %d\n',LM);
%number of meassurements
%M = 2;

%stepsize beta
%for LM beta will be substituted 
beta = 0.01;

if source
    
    Nu = input('please enter number of inputs of the MLP: ');
    N = input('please enter number of hidden-neurons of the MLP: ');
    Ny = input('please enter number of outputs of the MLP: ');
    
    M = input('please enter number of meassurements M: ');
    
    %parameters of hidden-neuron layer
    W_1 = zeros(N,Nu);
    %parameters of output-neuron layer
    W_2 = zeros(Ny,N);
    
    B_1 = zeros(N,M);
    B_2 = zeros(Ny,M);
    
    Y_p = zeros(Ny,M);
    Y = zeros(Ny,M);
    U = zeros(Nu,M);
else
    load(filename);
    
end%if(source)

E = zeros(Ny,M);

%parameter vektor
w = zeros(N*(Nu+Ny+1)+Ny,1);
%delta
Dw = zeros(N*(Nu+Ny+1)+Ny,1);
%vektor with partial differentation
dw = zeros(N*(Nu+Ny+1)+Ny,1);

if LM
   beta_LM = w * w';
   beta_LM_s = w;
end

if source
    %user initialization of start-parameters
    fprintf('\nplease enter parameter W_1\n');
    for posN = 1:N
        for posNu = 1:Nu
            fprintf('W_1(%d,%d)',posN,posNu);
            W_1(posN,posNu) = input(': ');
        end
    end
    
    fprintf('\nplease enter parameter W_2\n');
    for posNy = 1:Ny
        for posN = 1:N
            fprintf('W_2(%d,%d)',posNy,posN);
            W_2(posNy,posN) = input(': ');
        end
    end
    
    fprintf('\nplease enter parameter B_1\n');
    for posN = 1:N
        %for posM = 1:M
        fprintf('B_1(%d,%d)',posN,1);
        B_1(posN,1) = input(': ');
        %end
    end
    B_1 = B_1 * ones(M)';
    
    fprintf('\nplease enter parameter B_2\n');
    for posNy = 1:Ny
        %for posM = 1:M
        fprintf('B_2(%d,%d)',posNy,1);
        B_2(posNy,1) = input(': ');
        %end
    end
    B_2 = B_2 * ones(M)';
    
    fprintf('\nplease enter Y_p\n');
    for posNy = 1:Ny
        for posM = 1:M
            fprintf('Y_p(%d,%d)',posNy,posM);
            Y_p(posNy,posM) = input(': ');
        end
    end
    
    fprintf('\nplease enter U\n');
    for posNu = 1:Nu
        for posM = 1:M
            fprintf('U(%d,%d)',posNu,posM);
            U(posNu,posM) = input(': ');
        end
    end
end%if(source)


% init live subplot
figure;
hold on;
posSub = 1;
for posNy = 1:Ny
for m = 1:M
    %subplot(Ny,m,posNy);
    subplot(Ny,M,posSub);
    posSub = posSub + 1;
    hline(posNy,m) = stem(nan,nan);
end
end
w_hist = zeros(length(w),1);
count = 1;

%% computation of first model-output Y with start parameters

for m =1:M
    Y(:,m) = B_2(:,m) + W_2(:,:) * tanh(B_1(:,m) + W_1(:,:) * U(:,m));
end%for 1:M

%% computation of w_ideal
%error-matrix E
E = Y_p - Y;

%minimalization criterion
J = 0.5 * trace(E * E');
J_hist = J;
delta_J = 10;

%initialization of vektor w
%w = [ W_1(1,:) ... W_1(N,:) W_2(1,:) ... W_2(Ny,:) B_1' B_2' ]'
posw = 1;

%W_1 >> w
for posN = 1:N
    for posNu = 1:Nu
        w(posw,1) = W_1(posN,posNu);
        posw = posw + 1;
    end
end

%W_2 >> w
for posNy = 1:Ny
    for posN = 1:N
        w(posw,1) = W_2(posNy,posN);
        posw = posw + 1;
    end
end

%B_1 >> w
for posN = 1:N
    w(posw,1) = B_1(posN,1);
    posw = posw + 1;
end

%B_2 >> w
for posNy = 1:Ny
    w(posw,1) = B_2(posNy,1);
    posw = posw + 1;
end


while J > 1e-5 && count ~= 1000 && (abs(delta_J) > 5e-3);

%% computation of partial-differential-equations
%beta = abs(J);

for m = 1:M
    for posNy = 1:Ny
        %init dw with zeros <=> multiplication of Y with e_i'
        dw = zeros(length(w),1);
        
        %partial diff for B_2 
        dw(N*(Nu+Ny+1)+posNy,1) = 1;
        
        for posN = 1:N
            %partial diff W_2(posNy,:)'
            dw(N*Nu + (posNy-1)*N + posN,1) = tanh( B_1(posN,1) + W_1(posN,:) * U(:,m) );
            %(N*Nu + (posNy-1)*N,1)
            %partial diff B_1
            dw(N*(Nu+Ny) + posN,1) = W_2(posNy,posN) * (1 - dw(N*Nu + (posNy-1)*N,1)^2 ) * 1;
            
            for posNu = 1:Nu
                %partial diff W_1(posN,:)'
                dw((posN-1)*Nu + posNu ,1) = dw( N*(Nu+Ny) + posN,1) * U(posNu,m);               
            end%for(posNu)
        end%for(posN)
        
        %vgl. eq. (40) sum(i=1,Ny)
        Dw = Dw + dw * E(posNy,m);
        
        %Levenberg-Marquardt for calculation of beta-equivalent
        if LM
            beta_LM = beta_LM + dw * dw'; 
        end
        
    end%for(posNy)
    
    if LM
        %beta_LM = beta_LM_s * beta_LM_s';
        if det(beta_LM) < (10e5 * eps)
            mue = 10;
        else
            mue = 0;
        end
        
        beta = inv(beta_LM + mue * eye(length(dw)));
    end
    
    Dw = beta * Dw;
    
    w = w + Dw;
    w_hist = [w_hist w];
end%for(M)

%% w  >> W_1 W_2 B_1 B_2
posw = 1;
%W_1 << w
for posN = 1:N
    for posNu = 1:Nu
        W_1(posN,posNu) = w(posw,1);
        posw = posw + 1;
    end
end

%W_2 << w
for posNy = 1:Ny
    for posN = 1:N
        W_2(posNy,posN) = w(posw,1);
        posw = posw + 1;
    end
end

%B_1 << w
for posN = 1:N    
    for m = 1:M
        B_1(posN,m) = w(posw,1);
    end
    posw = posw + 1;
end


%B_2 << w
for posNy = 1:Ny
    for m = 1:M
        B_2(posNy,m) = w(posw,1);
    end
    posw = posw + 1;
end



for m =1:M
    Y(:,m) = B_2(:,m) + W_2(:,:) * tanh(B_1(:,m) + W_1(:,:) * U(:,m));
end%for 1:M

%error-matrix E
E = Y_p - Y;

%minimalization criterion
J = 0.5 * trace(E * E');
J_hist = [J_hist, J];

if (count > 100)
    delta_J = J - J_hist(count-90);
end

posSub = 1;
for posNy = 1:Ny
for m = 1:M
    %subplot(Ny,m,posNy);
    subplot(Ny,M,posSub);
    %gcf;
    posSub = posSub + 1;
    %plot(1:count,Y_p(posNy,1),1:count,Y(posNy,1));
    set(hline(posNy,m),'XData',count);
    set(hline(posNy,m),'YData',Y(posNy,m));
    set(hline(posNy,m),'Marker','square');        
    Y_p_plot((posNy*M - M + m ),count) = Y_p(posNy,m); %Y_p_plot((posNy,count))=
    Y_plot((posNy*M - M + m ),count) = Y(posNy,m);
    drawnow;
end
end
count = count + 1;


end %while(J>1e-4)

figure;
posSub = 1;
for posNy = 1:Ny
for m = 1:M
    subplot(Ny,M,posSub);

    posSub = posSub + 1;
    plot(1:count-1,Y_p_plot((posNy*M - M + m ),:),1:count-1,Y_plot((posNy*M - M + m ),:));
    xlabel('Step counter');
    ylabel({'Ny = ',posNy});
    title({'Meassurement = ',m});
end
end





