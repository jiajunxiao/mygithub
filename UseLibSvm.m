clc,clear,close all
load fisheriris % 导入数据集，fisheriris是Matlab自带的数据集
% 选择两类进行分类
feature=meas(51:end,1:2);  %取特征数据的1，2列特征作为分类特征
%feature=meas(51:end,3:4); %取特征数据的3，4列特征作为分类特征
groupcell = species(51:end);  %选取分类的类别
% 将类别置换为0，1
group=ones(100,1);
for i=1:100
    if(strcmp('versicolor',groupcell(i,1)))
        group(i,1)=1;
    else
        group(i,1)=0;
    end
end
p=10;
%评价指标向量
ACC_V=zeros(1,p);
ERR_V=zeros(1,p);
EER_V=zeros(1,p);
PRE_V=zeros(1,p);
REC_V=zeros(1,p);
F1_V=zeros(1,p);
AUC_V=zeros(1,p);
Score=[];
%交叉验证
data=crossvalind('Kfold',group,p); %选择样本中10%作为测试项，90%作为训练项
for k=1:p %交叉验证k=10，10个包轮流作为测试集
    test=(data==k); % 获得测试集元素在数据集中对应的单元编号
    train=~test; % 训练集元素的编号为非测试集元素的编号
    feature_train=feature(train,:);
    feature_test=feature(test,:);
    group_train=group(train,:);
    group_test=group(test,:);
    
    %训练模型并预测
    model =svmtrain(group_train,feature_train,'-b 1  -t 2');
    [PredictLabel,accuracy,s] = svmpredict(group_test,feature_test,model,'-b 1');
    
    %cop=[s(:,1) PredictLabel];
    cop=[s(:,1) group_test];
    Score=[Score;cop];
    
    %自带绘制ROC曲线
    figure;
    plotroc(group_test',s(:,1)');
%     com_score=[PredictLabel s(:,1)];
%     group_score=sortrows(com_score,2);
%     index=[1:length(PredictLabel)];
%     index=index';
%     group_score=[group_score index];
%     %auc=AUC(group_score);

    %绘图
    figure;
    clf;
    hold on;
    grid on;
    h1 = plot(feature(1:50,1),feature(1:50,2),'r+');
    h2 = plot(feature(51:100,1),feature(51:100,2), 'g*');
    h3 = plot( model.SVs(:,1),model.SVs(:,2),'o' );
    legend([h1,h2,h3],'class1','class2','Support Vectors');
    xlabel('demension1','FontSize',12);
    ylabel('demension2','FontSize',12);
    title('The visualization of classification','FontSize',12);

    [rolconMa1t,order] = confusionmat(group_test,PredictLabel); % t计算混淆矩阵
    %变换混淆矩阵
    conMalt=zeros(2,2);
    conMalt(1,1)=rolconMa1t(2,2);
    conMalt(1,2)=rolconMa1t(2,1);
    conMalt(2,1)=rolconMa1t(1,2);
    conMalt(2,2)=rolconMa1t(1,1);

    %---------
    %真正率
    TPR=conMalt(1,1)/(conMalt(1,1)+conMalt(1,2));
    %真负率
    TNR=conMalt(2,2)/(conMalt(2,2)+conMalt(2,1));
    %假正率(FAR)
    FPR=conMalt(2,1)/(conMalt(2,1)+conMalt(2,2));
    %假负率(FFR)
    FNR=conMalt(1,2)/(conMalt(1,2)+conMalt(1,1));
    %---------

    %阈值变化
    theat=linspace(0,1,99);
    %指标的矩阵
    PRE=zeros(1,99);
    FFR=zeros(1,99);
    FAR=zeros(1,99);
    TPR=zeros(1,99);
    zhongjian=[];
    REC=zeros(1,99);
    F1=zeros(1,99);
    %等错误率
    EER=0;
    newPredicLabel=zeros(10,1);
    for indtheat=1:length(theat)
        for index=1:length(PredictLabel)
            if(s(index,1)>theat(indtheat))
                newPredictLabel(index,1)=1;
            else
                newPredictLabel(index,1)=0;
            end
        end
        [rolconMa1tinner,order1] = confusionmat(group_test,newPredictLabel); % t计算混淆矩阵   
        conMa1tinner=zeros(2,2);
        conMa1tinner(1,1)=rolconMa1tinner(2,2);
        conMa1tinner(1,2)=rolconMa1tinner(2,1);
        conMa1tinner(2,1)=rolconMa1tinner(1,2);
        conMa1tinner(2,2)=rolconMa1tinner(1,1);
        %pre指标（精确率）
        pre=conMa1tinner(1,1)/(conMa1tinner(1,1)+conMa1tinner(2,1));
        newpre=isnan(pre);
        if(newpre)
            pre=newpre;
        end
        %rec指标（召回率）
        rec=conMa1tinner(1,1)/(conMa1tinner(1,1)+conMa1tinner(1,2));
        % F1指标
        f1=(2*pre*rec)/(pre+rec);
        %真正率
        tpr=conMa1tinner(1,1)/(conMa1tinner(1,1)+conMa1tinner(1,2));
        %假正率(FAR,FPR)
        far=conMa1tinner(2,1)/(conMa1tinner(2,1)+conMa1tinner(2,2));
        %假负率(FFR)
        ffr=conMa1tinner(1,2)/(conMa1tinner(1,2)+conMa1tinner(1,1));
        FAR(1,indtheat)=far;
        FFR(1,indtheat)=ffr;
        TPR(1,indtheat)=tpr;
        PRE(1,indtheat)=pre;
        REC(1,indtheat)=rec;
        F1(1,indtheat)=f1;
    end
    
    %PR曲线
    figure;
    clf;
    hold on;
    plot(REC,PRE,'r');
    axis([0,1,0,1]);
    set(gca,'xtick',[0:0.1:1]);  
    set(gca,'ytick',[0:0.1:1]);  
    xlabel('REC','FontSize',12);
    ylabel('PRE','FontSize',12);
    title('PR曲线','FontSize',12);

    [xzeros,yzeros,b]=EERpoint(theat,FAR,FFR);
    xzeros(find(xzeros==0))=[];
    %yzeros(find(yzeros==0))=[];
    indx=find(~isnan(xzeros));
    if(yzeros(indx(1))==0)
        EER=FAR(indx(1));
    else
        EER=yzeros(indx(1));
    end
    %FAR,FFR以及等错误率
    figure;
    clf;
    hold on;
    [AX]=plotyy(theat,FAR,theat,FFR);
    plot(theat(indx(1)),EER,'o');
    %plot(theat,FFR,'g');
    axis([0,1,0,1]);
    set(gca,'xtick',[0:0.1:1]);  
    set(AX(1),'ytick',[0:0.05:1]); 
    set(AX(2),'ytick',[0:0.05:1]); 
    xlabel('theat','FontSize',12);
    ylabel('FAR','FontSize',12);
    set(get(AX(2),'ylabel'),'string','FFR');

    % ROC曲线
    figure;
    clf;
    hold on;
    plot(FAR,TPR,'r');
    plot(FAR,FAR,'b');
    plot(FAR,1-FAR,'g');
    axis([0,1,0,1]);
    set(gca,'xtick',[0:0.1:1]);  
    set(gca,'ytick',[0:0.1:1]);
    xlabel('FPR','FontSize',12);
    ylabel('TPR','FontSize',12);
    title('ROC曲线','FontSize',12);
    % 计算ROC曲线线下面积
    auc=abs(trapz(FAR,TPR));
    AUC_V(1,k)=auc;
    
%   %ERR指标（错误率）
%   ERR=(conMalt(2,1)+conMalt(1,2))/(sum(sum(conMalt)));
    %ACC指标（精准率）
    ACC=(conMalt(1,1)+conMalt(2,2))/(sum(sum(conMalt)));
    %pre指标（精确率）
    PREC=conMalt(1,1)/(conMalt(1,1)+conMalt(2,1));
    %rec指标（召回率）
    RECA=conMalt(1,1)/(conMalt(1,1)+conMalt(1,2));
    % F1指标
    actF1=(2*PREC*RECA)/(PREC+RECA);
    ACC_V(1,k)=ACC;
    EER_V(1,k)=EER;
    PREC_V(1,k)=PREC;
    RECA_V(1,k)=RECA;
    actF1_V(1,k)=actF1;

end



%平均ACC,EER,PREC,REC,F1的结果
act_ACC=sum(ACC_V)/p;
act_EER=sum(EER_V)/p;
act_PRE=sum(PREC_V)/p;
act_REC=sum(REC_V)/p;
act_F1=sum(actF1_V)/p;
act_AUC=sum(AUC_V)/p;
%导出数据
dlmwrite('svm_score.txt', Score, ' ');

function [xzero,yzero,y]=EERpoint(x,y1,y2)
y=y1-y2;
nLen=length(x);
xzero=zeros(1,nLen);
yzero=zeros(1,nLen);
for i=1:nLen-1
    if y(i)*y(i+1)==0   %等于0的情况
        if y(i)==0
            xzero(i)=i;
            yzero(i)=0;
        end
        if y(i+1) == 0
            xzero(i+1)=i+1;
            yzero(i+1)=0;
        end
    elseif y(i)*y(i+1)<0  
        %一定有交点，用一次插值
        k = abs(y(i))/(abs(y(i))+abs(y(i+1)));%交点在i与i+1之间的比例
        xzero(i)=i + k;
        yzero(i)=y1(i)+ (y1(i+1) - y1(i))*k;
    else            
    end
    if (xzero(i)==0)&(yzero(i)==0)     %除掉不是交点的部分
        xzero(i)=nan;
        yzero(i)=nan;
    end
end
end