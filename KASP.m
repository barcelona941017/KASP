function [acc,dice,ri,runtime] = KASP(filename,gTruth_filename,nbSegment,k,img_type)

filename = 'D:\学习\学术\快速谱聚类算法的研究\MFSC\对比实验\KASP\color_img\b17paul1444.png';
gTruth_filename = 'D:\学习\学术\快速谱聚类算法的研究\MFSC\对比实验\KASP\golden2Obj256\b17paul1444_golden.jpg';
nbSegment = 5;
k = 150;
img_type = 2;
tic;
I = imread(filename);
[m, n, p] = size(I);
I_reshape = reshape(I,m*n,p);
I_reshape = double(I_reshape);
%I_reshape_posi = [c I_reshape];
[I_seg,Central] = kmeans(I_reshape,k);
trans_matrix = sparse(k,m*n);
for i = 1:k
    index = find(I_seg == i);
    trans_matrix(i,index) = 1;
end



%I_seg = reshape(I_seg,m,n);

[W,Dist] = compute_relation(Central');                                    %代表点的W 输入需要列表示个数， 行表示特征
dataNcut.valeurMin=1e-6;
W = sparsifyc(W,dataNcut.valeurMin);

%[IndiMat, order_of_node] = reprePointIndictorMatrix(data, W, Q_IndexTable, kdTree_stru, total_leaf_node_index,start_level, nbCluster);   %指示向量
%IndiMat_sparse = sparse(IndiMat);

%total_trans_Matrix = total_trans_Matrix(order_of_node,:);
%W = W(order_of_node,order_of_node);
d = sum(abs(W),2);
n_2 = size(W,1);
W = spdiags(d,0,n_2,n_2) - W;
D = 1./sqrt(d+eps);

PNew = spmtimesd(W,D,D);
[eigenVector, s] = eigs(PNew, nbSegment,'sm');
s = real(diag(s));
[x,y] = sort(-s); 
Eigenvalues = -x;
[row,col] = size(eigenVector);
eigenVector2 = eigenVector.*eigenVector;
totalQ = sum(eigenVector2,2);
for i =1:row  
    eigenVector(i,:) = eigenVector(i,:)/sqrt(totalQ(i));
end

[centers,U] = fcm(eigenVector,nbSegment );
SegLabel = zeros(1,n_2);
maxU = max(U);
for ii = 1 :nbSegment
   SegLabel(U(ii,:)== maxU) = ii;
end

I_seg = SegLabel * trans_matrix;
I_seg = reshape(I_seg,m,n);
imagesc(I_seg);
showSegLabel = (255/nbSegment) * I_seg;
showSegLabel = uint8(showSegLabel);
toc
figure(11),
imshow(showSegLabel);
imwrite(showSegLabel, 'C:\Users\Administrator\Desktop\毕业论文\算法会用到的图片\四叉树算法\gray.png');
[ACC,RI,DICE] = showACCandSaveRes(I,I_seg,gTruth_filename,img_type, nbSegment);
