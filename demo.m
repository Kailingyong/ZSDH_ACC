
close all; clear all; 
clc;

options.choice = 'evaluation';

addpath('./utils/');

%% load data
load './data/LabelmeZeroShot.mat'

%load lableme;
fprintf('Labelme dataset loaded...\n');
 rand('seed',1);

% %% centralization
fprintf('centralizing data...\n');
I_te = bsxfun(@minus, I_te, mean(I_tr, 1)); I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
T_te = bsxfun(@minus, T_te, mean(T_tr, 1)); T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

% %% kernelization
fprintf('kernelizing...\n\n');
[I_tr1,I_te1]=Kernelize(I_tr,I_te); [T_tr1,T_te1]=Kernelize(T_tr,T_te);
I_te = bsxfun(@minus, I_te1, mean(I_tr1, 1)); I_tr = bsxfun(@minus, I_tr1, mean(I_tr1, 1));
T_te = bsxfun(@minus, T_te1, mean(T_tr1, 1)); T_tr = bsxfun(@minus, T_tr1, mean(T_tr1, 1));



L_tr_Matrix = L_tr;
L_te_Matrix = L_te;

[a  L_tr] = max(L_tr');
[a  L_te] = max(L_te');

L_tr = L_tr';
L_te = L_te';

topK = 1:1000;
run = 10;


for i = 1 : run
    tic
     rand('seed',1);
classes = randperm(8);
seenClass = classes(3:end);
unseenClass = classes(1:2);
fprintf('Unseen classes:\n');
 unseenClass
 
 temp = zeros(length(L_te),1);
for ii=1:2
    temp = temp + ismember(L_te,unseenClass(ii));
end
index_unseen_in_te = find(temp==1);
index_seen_in_te = [1:length(L_te)]';
index_seen_in_te(index_unseen_in_te) = [];
% ------------------------------------------
temp = zeros(length(L_tr),1);
for ii=1:2
    temp = temp + ismember(L_tr,unseenClass(ii));
end
index_unseen_in_tr = find(temp==1);
index_seen_in_tr = [1:length(L_tr)]';
index_seen_in_tr(index_unseen_in_tr) = [];


%%

% train data of seen class. same as retrieal data
X1_SR = I_tr(index_seen_in_tr,:);
X2_SR = T_tr(index_seen_in_tr,:);
L_SR = L_tr_Matrix(index_seen_in_tr,:);
S_SR= S_tr(index_seen_in_tr,:);

X1_SQ = I_te(index_seen_in_te,:);
X2_SQ = T_te(index_seen_in_te,:);
L_SQ = L_te_Matrix(index_seen_in_te,:);

% data split of unseen data
X1_UR = I_tr(index_unseen_in_tr,:);
X2_UR = T_tr(index_unseen_in_tr,:);
L_UR = L_tr_Matrix(index_unseen_in_tr,:);

X1_UQ = I_te(index_unseen_in_te,:);
X2_UQ = T_te(index_unseen_in_te,:);
L_UQ = L_te_Matrix(index_unseen_in_te,:);
                                                                                                                                                                                                                                                              

S = labelme_attributes(seenClass,:);


%% ²ÎÊýÉèÖÃ
lambda1 = 1;  lambda2 = 1;
mu = 1e-3; beta = 1e3;
alphe =1e4; gamma = 1e4; thea = 1e-5;


 
bits = 64 ;
numiter = 10;

%% Learn ZSDH_ACC
[ W1, W2, P, t1, t2, B2, B,Z,obj] = ZSDH_ACC(X1_SR, X2_SR, L_SR(:,seenClass),S, gamma, numiter, bits, lambda1, lambda2, mu, alphe, beta,thea);


%% Compute mAP
 
    rBX = [sign(P * Z *(W1'*X1_UR' + t1*ones(1,size(X1_UR,1))))';B'];
    qBX = sign(P * Z *(W1'*X1_UQ' + t1*ones(1,size(X1_UQ,1))))';
    rBY = [sign(P * Z *(W2'*X2_UR' + t2*ones(1,size(X2_UR,1))))';B'];
    qBY = sign(P * Z *(W2'*X2_UQ' + t2*ones(1,size(X2_UQ,1))))';

    rBX = (rBX > 0);
    qBX = (qBX > 0);
    rBY = (rBY > 0);
    qBY = (qBY > 0);
    
    fprintf('\nText-to-Image Result:\n');
    mapTI = map_rank1(qBY, rBX,[L_UR; L_SR], L_UQ);
    map(i,1) = mapTI(end);
    
    
    fprintf('Image-to-Text Result:\n');
    mapIT = map_rank1(qBX, rBY,[L_UR; L_SR], L_UQ);
    map(i,2) = mapIT(end);
   
end
 mean(map)
