function [ W1, W2, P, t1, t2, B1, B,Z,obj] = ZSDH_ACC(X1, X2, Ltrain, A1, gamma, numiter, bits, lambda1, lambda2, mu, alphe,beta,thea)


%% æÿ’Û≥ı ºªØ
A1 = A1';
W1 = randn(size(A1,1),size(X1,2));
W2 = randn(size(A1,1),size(X2,2));
t1 = randn(size(A1,1),1); 
t2 = randn(size(A1,1),1);


P = randn(bits,size(A1,1));
B1 = sign(randn(bits,size(X1,1)));
B = sign(randn(bits,size(X1,1)));
L_tr = Ltrain;
Ltrain = Ltrain';

% Calculate A
for i= 1:size(Ltrain,2)
    A(:,i) = (A1*Ltrain(:,i))./sum(Ltrain(:,i));
end


m1 = size(X1,2);
m2 = size(X2,2);
m = size(X2,1);
e = ones(1,size(X1,1));
X1 = X1';
X2 = X2';
obj = [];
%% Update Parameters
for i = 1:numiter
    
    % update W1 and W2    
    W1 = pinv(lambda1*X1*X1'+ gamma*eye(m1))*(lambda1*X1*(A-t1*e)');
    W2 = (lambda2*X2*X2'+ gamma*eye(m2))\(lambda2*X2*(A-t2*e)');

    % update t1 and t2     
    t1 = ((A - W1'*X1)*e')/m;
    t2 = ((A - W2'*X2)*e')/m;
     
    % update Z
    Z1 = pinv(mu*P'*P + beta * eye(size(A,1)))*(beta*A*A' + mu*P'*B1*A');
    Z = Z1*pinv((beta+mu)*A*A');
  
    
    % update P
    P = (mu*B1*A'*Z')*pinv(mu*Z*A*A'*Z' + gamma*eye(size(A,1)));
    

%%B1
  Q1 = alphe*bits*B*L_tr*Ltrain + mu*P*Z*A + thea*B;    
for time = 1:10  
   Z0 = B1;
    for k = 1 : size(B1,1)
        Zk = B1'; Zk(:,k) = [];
        Wkk = B(k,:); Wk = B; Wk(k,:) = [];                    
        B1(k,:) = sign(Q1(k,:) -  alphe*Wkk*Wk'*Zk');
    end

    if norm(B1-Z0,'fro') < 1e-6 * norm(Z0,'fro')
        break
    end
end



%%B
  Q = alphe*bits*B1*Ltrain'*L_tr' + thea*B1;
for time = 1:10           
   Z0 = B;
    for k = 1 : size(B,1)
        Zk = B'; Zk(:,k) = [];
        Wkk = B1(k,:); Wk = B1; Wk(k,:) = [];                    
        B(k,:) = sign(Q(k,:) -  alphe*Wkk*Wk'*Zk');
    end

    if norm(B-Z0,'fro') < 1e-6 * norm(Z0,'fro')
        break
    end
end


    
    % compute objective function
    norm1 = lambda1*norm(A - W1' * X1 - t1 * e, 'fro') + lambda2*norm(A - W2' * X2 - t2 * e, 'fro');
    norm2 = beta*norm(A - Z*A, 'fro');
    norm3 = mu*norm(B1 - P*Z*A, 'fro');
    norm5 = thea*norm(B1 - B, 'fro');
    norm6 = gamma * (norm(W1, 'fro') + norm(W2, 'fro') + norm(P, 'fro'));
    currentF= norm1 + norm2 + norm3 + norm5 + norm6;
    disp(currentF);
    obj = [obj,currentF];
end

end
