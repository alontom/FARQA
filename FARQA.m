function [coefsSD, lam, coefsDiag, diagRateAvg, coefsWKT, dfas]=FARQA(TS,tau,dim,r,norm,zscore)
% Required functions:
% dfa - Weron, R. (2011). DFA: MATLAB function to compute the Hurst exponent using Detrended Fluctuation Analysis (DFA).
% MDRQA - Wallot, S., Roepstorff, A., & MÃ¸nster, D. (2016). Multidimensional Recurrence Quantification Analysis (MdRQA) for the analysis of
% multidimensional time-series: A software implementation in MATLAB and its application to group-level data in joint action. Frontiers in Psychology,
% 7, 1835. http://dx.doi.org/10.3389/fpsyg.2016.01835

% Inputs:
% TS - time series (a column vector)
% tau - delay parameter
% dim - embedding parameter
% r - threshold parameter
% norm - the type of norm by with the phase-space is normalized. The
%    following norms are available:
%    'euc' - Euclidean distance norm
%    'max' - Maximum distance norm
%    'min' - Minimum distance norm
%    'non' - no normalization of phase-space
% zscore - whether the data should be z-scored
%    before performing RQA:
%    0 - no z-scoring of TS
%    1 - z-score columns of TS

% Outputs:
% coefsSD - Approach 1: SD(%REC) over bin size - including coef, p-val, and R2
% lam - Approach 2: Laminarity
% coefsDiag - Approach 3: Diag %REC over diag index - including coef, p-val, and R2
% diagRateAvg - Approach 4: Diag ratio
% coefsWKT - Wiener Khinchin Theorem: FT of Diag %REC over diag index - including coef, p-val, and R2
% dfas - H exponent, Detrended Fluctuation Analysis (DFA)

% The authors give no warranty for the correct functioning of the software
% and cannot be held legally accountable.

sigLen=length(TS); % Signal's length
coefsSD=nan(1,3); % Approach 1: SD(%REC) over bin size - including coef, p-val, and R2
lam=nan(1); % Approach 2: Laminarity
coefsDiag=nan(1,3); % Approach 3: Diag %REC over diag index - including coef, p-val, and R2
diagRate=nan(1,(sigLen)/2);
diagRateAvg=nan(1); % Approach 4: Diag ratio
WKT=nan(1,sigLen);
coefsWKT=nan(1,3); %Wiener Khinchin Theorem: FT of Diag %REC over diag index - including coef, p-val, and R2
dfas=nan(1); % Detrended Fluctuation Analysis (DFA)

[RP,tmpRes]=MDRQA(TS,dim,tau,norm,r,zscore); 
% Analysis 1: REC std over different bin sizes
lenRP=length(RP);
RECsz=[];
szs=2.^(1:15);
szs=szs(szs<lenRP/2);
for sz=szs
    tmpREC=[];
    for indx=1:sz:lenRP-sz
        for indy=1:sz:lenRP-sz
            tmpREC=[tmpREC,sum(sum(RP(indx:indx+sz-1,indy:indy+sz-1)))/sz^2];
        end
    end
    RECsz=[RECsz,std(tmpREC)];
end
md1=fitlm(log2(szs(1,1:end-1))',log2(RECsz(1,1:end-1))');
coefsSD(1,1)=md1.Coefficients(2,1).Estimate;
coefsSD(1,2)=md1.Coefficients(2,4).pValue;
coefsSD(1,3)=md1.Rsquared.Adjusted;
% Analysis 2: Laminarity
lam(1)=tmpRes(7);
% Analysis 3: Diangonal REC%
dREC=nan(1,(lenRP));
for di=0:lenRP-1 % 0=main diagonal
    curDiag=diag(imrotate(RP,-90),di);
    dREC(di+1)=sum(curDiag)/length(curDiag);
end
md3=fitlm(log2(1:64),log2(dREC(2:65))); % Picking 64 first values (except main diagonal)
coefsDiag(1,1)=md3.Coefficients(2,1).Estimate;
coefsDiag(1,2)=md3.Coefficients(2,4).pValue;
coefsDiag(1,3)=md3.Rsquared.Adjusted;
%Analysis 4: Subsequent Diagonal Ratio 
diagRate(1,1:lenRP/2)=dREC(2:2:(lenRP))./dREC(1:2:(lenRP));
diagRateAvg(1)=mean(diagRate(1,2:65));
% Analysis 5: Wiener Khinchin Theorem
WKT(1,1:lenRP)=abs(fft(dREC));
md4WKT=fitlm(log2(1:64),log2(WKT(1,2:65)));
coefsWKT(1,1)=md4WKT.Coefficients(2,1).Estimate;
coefsWKT(1,2)=md4WKT.Coefficients(2,4).pValue;
coefsWKT(1,3)=md4WKT.Rsquared.Adjusted;
% Analysis 6: DFA
dfas(1)=dfa(TS);
end


