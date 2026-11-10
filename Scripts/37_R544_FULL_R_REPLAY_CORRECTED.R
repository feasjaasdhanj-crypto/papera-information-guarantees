# Paper A R544: pure-base-R cold-start replay of EEG feature-level analysis.
# Scientific input: 106-subject R539 derived EEG features (not raw EDF).
# No network calls, no Python, no external R packages. Rscript >=4.2 suggested.
# All tests are subject-disjoint; training ranking, validation selection, held-out subject evaluation.
options(stringsAsFactors=FALSE, scipen=999)
args <- commandArgs(trailingOnly=TRUE)
base <- if(length(args)) normalizePath(args[1L],winslash='/',mustWork=TRUE) else getwd()
ind <- file.path(base,'R544_REPRO_INPUT')
out <- file.path(base,'R544_REPRO_OUTPUT')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
logfile <- file.path(out,'R544_R_EXECUTION_STATUS.txt')
cat('START',format(Sys.time()),'\n',file=logfile)
logit <- function(...) {a <- paste(...,collapse=' ');cat(a,'\n');cat(a,'\n',file=logfile,append=TRUE)}
# Errors MUST be visible to the operator and retained in the return ZIP.
options(error=function() {
  message('[FAIL] R full replay aborted: ',geterrmessage())
  cat('FAIL: ',geterrmessage(),'\n',file=logfile,append=TRUE)
  traceback(4)
  quit(save='no',status=1L,runLast=FALSE)
})
logit('[START] Model replay from extracted EEG features; all five subject folds and two calibration protocols')


# Reproduce sklearn LDA(solver="lsqr", shrinkage="auto", priors=[1/3]*3).
# Ledoit-Wolf uses within-class standardized observations and population covariances.
pop_sd <- function(x) sqrt(colMeans(sweep(x,2L,colMeans(x),'-')^2))
lda_fit <- function(X, y) {
  p <- ncol(X); classes <- 0:2
  means <- matrix(0.0,3L,p); covp <- matrix(0.0,p,p)
  for(k in 1:3) {
    G <- X[y == classes[k],,drop=FALSE]
    if(nrow(G)<3) stop('Insufficient data within class')
    means[k,] <- colMeans(G)
    centered <- sweep(G,2L,means[k,],'-')
    scales <- sqrt(colMeans(centered^2));scales[scales<1e-15] <- 1
    Z <- sweep(centered,2L,scales,'/')
    n <- nrow(Z)
    emp <- crossprod(Z)/n
    mu <- sum(diag(emp))/p
    Z2 <- Z^2
    beta_sum <- sum(crossprod(Z2))
    delta_sum <- sum(crossprod(Z)^2)/(n^2)
    beta <- (beta_sum/n-delta_sum)/(p*n)
    delta <- (delta_sum-2*mu*sum(diag(emp))+p*mu^2)/p
    shrinkage <- if(delta<1e-16) 0 else min(beta,delta)/delta
    cstd <- (1-shrinkage)*emp+shrinkage*mu*diag(p)
    craw <- cstd*outer(scales,scales)
    covp <- covp+craw/3
  }
  coef <- t(solve(covp,t(means)))
  intercept <- -0.5*rowSums(means*coef)+log(1/3)
  list(coef=coef,intercept=intercept)
}
lda_predict <- function(model,X) {
  score <- sweep(X%*%t(model$coef),2,model$intercept,'+')
  score <- sweep(score,1,apply(score,1,max),'-')
  pp <- exp(score)
  sweep(pp,1,rowSums(pp),'/')
}
loss <- list(
 balanced=matrix(c(0,1,1,1,0,1,1,1,0),nrow=3,byrow=TRUE),
 miss_T2_x3=matrix(c(0,1,1,1,0,1,3,3,0),nrow=3,byrow=TRUE),
 false_alarm_T2_x3=matrix(c(0,1,3,1,0,3,1,1,0),nrow=3,byrow=TRUE),
 reject035=matrix(c(0,1,1,.35,1,0,1,.35,1,1,0,.35),nrow=3,byrow=TRUE))

# Subject-level equal-state risk; each participant has one value per measurement policy.
per_subject_risk <- function(probs,y,s,subjects,L) {
  action <- max.col(-probs%*%L,ties.method='first')
  vapply(subjects,function(u) {
    pos <- s==u
    mean(vapply(0:2,function(c) {
      ids <- which(pos & y==c)
      if(!length(ids)) stop('Missing state in held-out subject')
      mean(L[cbind(rep(c+1L,length(ids)),action[ids])])
    },numeric(1L)))
  },numeric(1L))
}

D <- read.csv(gzfile(file.path(ind,'EEG_106_SUBJECTS_ALL_EXTRACTED_FEATURES.csv.gz')),check.names=FALSE)
stopifnot(nrow(D)==9536L,length(unique(D$subject))==106L)
E <- c('FC5','FC3','FC1','C5','C3','C1','CP5','CP3')
R <- c('FC6','FC4','FC2','C6','C4','C2','CP6','CP4')
B <- c('theta','alpha','beta')
cols <- unlist(lapply(c(E,R),function(ele) paste(ele,B,sep='_')))
stopifnot(all(cols%in%names(D)), !anyNA(D[,cols]))
Y <- match(D$state,c('T0','T1','T2'))-1L
S <- D$subject
FOLD <- read.csv(file.path(ind,'R540_calibrated_subject_map.csv'))
stopifnot(setequal(unique(S),FOLD$subject))
folds <- lapply(0:4,function(f) FOLD$subject[FOLD$fold==f])
contracts <- names(loss)
all_policy <- list();all_per <- list();all_masks <- list();checks <- list();tie_records <- list()
target_key <- function(risk,mask) order(risk,mask)
for(proc in c('calibrated','global_train_only')) {
  X <- as.matrix(D[,cols]); storage.mode(X) <- 'double'
  if(proc=='calibrated') {
    # Unlabeled full-subject-session normalization; used only for measurements, not state labels.
    for(s in unique(S)) {
      ix <- which(S==s); block <- X[ix,,drop=FALSE]; mean_s <- colMeans(block)
      d0 <- sweep(block,2,mean_s,'-'); sd_s <- sqrt(colSums(d0^2)/(nrow(block)-1L))
      X[ix,] <- sweep(d0,2,pmax(sd_s,.05),'/')
    }
  }
  info <- read.csv(file.path(ind,paste0('R540_',proc,'_information.csv')))
  archived <- read.csv(file.path(ind,paste0('R540_',proc,'_portfolios.csv')))
  for(fold in 0:4) {
    logit('[PROGRESS] Starting',proc,'fold',fold+1,'of 5')
    testsubs <- folds[[fold+1L]]
    valsubs <- folds[[((fold+1L)%%5L)+1L]]
    trainsubs <- sort(setdiff(unique(S),c(testsubs,valsubs)))
    tr <- S%in%trainsubs;va <- S%in%valsubs;te <- S%in%testsubs
    stopifnot(!any(tr&va),!any(tr&te),!any(va&te))
    mu <- colMeans(X[tr,,drop=FALSE]);d0 <- sweep(X[tr,,drop=FALSE],2,mu,'-')
    sd <- sqrt(colMeans(d0^2));sd[sd<.01] <- 1
    xx <- sweep(sweep(X,2,mu,'-'),2,sd,'/')
    subinfo <- info[info$fold==fold,,drop=FALSE]
    subinfo <- subinfo[order(subinfo$index),,drop=FALSE]
    arch <- archived[archived$fold==fold,,drop=FALSE]
    trainrisk <- matrix(NA_real_,256L,length(contracts),dimnames=list(NULL,contracts))
    valrisk <- trainrisk
    testrisk <- trainrisk
    personrisk <- vector('list',256L)
    maxerr <- 0
    for(mask in 0:255) {
      if(mask%%32L==0L) logit('[PROGRESS]',proc,'fold',fold+1,'of 5, mask',mask,'of 255')
      sel <- ifelse(bitwAnd(mask,bitwShiftL(1L,0:7))!=0L,R,E)
      usecols <- unlist(lapply(sel,function(ele) paste(ele,B,sep='_')))
      ci <- match(usecols,cols)
      model <- lda_fit(xx[tr,ci,drop=FALSE],Y[tr])
      prtrain <- lda_predict(model,xx[tr,ci,drop=FALSE])
      prval <- lda_predict(model,xx[va,ci,drop=FALSE])
      prtest <- lda_predict(model,xx[te,ci,drop=FALSE])
      pps <- list()
      for(key in contracts) {
        rtrain <- per_subject_risk(prtrain,Y[tr],S[tr],trainsubs,loss[[key]])
        rval <- per_subject_risk(prval,Y[va],S[va],valsubs,loss[[key]])
        rt <- per_subject_risk(prtest,Y[te],S[te],testsubs,loss[[key]])
        trainrisk[mask+1L,key] <- mean(rtrain)
        valrisk[mask+1L,key] <- mean(rval)
        testrisk[mask+1L,key] <- mean(rt)
        pps[[key]] <- rt
        ref <- arch[arch$mask==mask,,drop=FALSE]
        err <- max(abs(mean(rval)-ref[[paste0('val_',key)]]),abs(mean(rt)-ref[[paste0('test_',key)]]))
        maxerr <- max(maxerr,err)
      }
      personrisk[[mask+1L]] <- pps
      all_masks[[length(all_masks)+1L]] <- data.frame(processing=proc,fold=fold,mask=mask,
        budget=sum(bitwAnd(mask,bitwShiftL(1L,0:7))!=0L),
        train_balanced=trainrisk[mask+1L,'balanced'],train_miss_T2_x3=trainrisk[mask+1L,'miss_T2_x3'],
        train_false_alarm_T2_x3=trainrisk[mask+1L,'false_alarm_T2_x3'],train_reject035=trainrisk[mask+1L,'reject035'])
    }
    checks[[length(checks)+1L]] <- data.frame(processing=proc,fold=fold,max_abs_archive_difference=maxerr)
    logit('Rebuilt',proc,'fold',fold,'archive_error',format(maxerr,digits=8))
    if(maxerr>1e-7) stop('HARD GATE: archive numerical mismatch >1e-7')
    for(budget in c(2L,4L,6L)) {
      available <- which(vapply(0:255,function(m)sum(bitwAnd(m,bitwShiftL(1L,0:7))!=0L),integer(1))==budget)-1L
      metrics <- sapply(available,function(m) {
        idx <- which(bitwAnd(m,bitwShiftL(1L,0:7))!=0L)
        c(full=sum(subinfo$full_delta[idx]),local=sum(subinfo$local_r2[idx]))
      })
      if(is.vector(metrics)) metrics <- matrix(metrics,nrow=2)
      rankfull <- available[order(metrics[1,],available)]
      ranklocal <- available[order(metrics[2,],available)]
      for(key in contracts) {
        ranktarget <- available[order(trainrisk[available+1L,key],available)]
        for(k in c(5L,10L)) {
          for(policy in c('target','full','local')) {
            ranking <- switch(policy,target=ranktarget,full=rankfull,local=ranklocal)
            shortlist <- ranking[seq_len(k)]
            # Cross-platform numerical tie convention: validation risks differing
            # by <= 1e-12 are equal for the purpose of selection. Choose the
            # lowest integer mask ID. No test labels or risks enter selection.
            validation_candidates <- valrisk[shortlist+1L,key]
            best_validation <- min(validation_candidates)
            near_best <- shortlist[validation_candidates <= best_validation + 1e-12]
            chosen <- min(near_best)
            if(length(near_best)>1L) {
              tie_records[[length(tie_records)+1L]] <- data.frame(
                processing=proc,fold=fold,budget=budget,contract=key,k=k,
                policy=policy,selected_mask=chosen,
                equivalent_masks=paste(sort(near_best),collapse=';'),
                validation_range=max(validation_candidates[shortlist%in%near_best])-
                  min(validation_candidates[shortlist%in%near_best]))
            }
            all_policy[[length(all_policy)+1L]] <- data.frame(processing=proc,fold=fold,budget=budget,
              contract=key,k=k,policy=policy,selected_mask=chosen,
              validation_risk=valrisk[chosen+1L,key],test_risk=testrisk[chosen+1L,key])
            risks <- personrisk[[chosen+1L]][[key]]
            all_per[[length(all_per)+1L]] <- data.frame(processing=proc,fold=fold,
              subject=testsubs,budget=budget,contract=key,k=k,policy=policy,
              selected_mask=chosen,risk=risks)
          }
        }
      }
    }
  }
}
policy <- do.call(rbind,all_policy)
per <- do.call(rbind,all_per)
write.csv(policy,file.path(out,'R544_R_recomputed_policy_choices.csv'),row.names=FALSE)
write.csv(per,file.path(out,'R544_R_recomputed_subject_risks.csv'),row.names=FALSE)
write.csv(do.call(rbind,checks),file.path(out,'R544_R_archive_QA.csv'),row.names=FALSE)
if(length(tie_records)>0L) write.csv(do.call(rbind,tie_records),
  file.path(out,'R544_R_numerical_ties.csv'),row.names=FALSE)

# Policy selection exact compare to the independently archived Python implementation.
expected <- read.csv(file.path(ind,'R544_equal_candidate_policy_choices.csv'))
merged <- merge(policy,expected,by=c('processing','fold','budget','contract','k','policy'),suffixes=c('_R','_locked'))
if(nrow(merged)!=nrow(expected) || !all(merged$selected_mask_R==merged$selected_mask_locked)) {
  write.csv(merged[merged$selected_mask_R!=merged$selected_mask_locked,,drop=FALSE],
    file.path(out,'R544_R_selection_mismatches.csv'),row.names=FALSE)
  stop('HARD GATE: candidate choice differs from locked archive even after numerical tie convention')
}
stopifnot(max(abs(merged$validation_risk_R-merged$validation_risk_locked))<1e-9,
  max(abs(merged$test_risk_R-merged$test_risk_locked))<1e-9)
logit('PASS: all',nrow(merged),'selected masks reproduce with stable numerical tie convention')

# Paired subject inference collapsed over 3 physical budgets; Holm adjusts 4 tasks x 2 shortlist sizes per baseline.
agg <- aggregate(risk~processing+subject+contract+k+policy,per,mean)
keys <- unique(agg[c('processing','contract','k')])
results <- list()
for(i in seq_len(nrow(keys))) {
  krow <- keys[i,];sub <- subset(agg,processing==krow$processing & contract==krow$contract & k==krow$k)
  tw <- subset(sub,policy=='target');tw <- tw[order(tw$subject),]
  for(ref in c('full','local')) {
    rr <- subset(sub,policy==ref);rr <- rr[order(rr$subject),]
    stopifnot(identical(tw$subject,rr$subject))
    diff <- tw$risk-rr$risk
    pv <- t.test(diff,mu=0,alternative='less')$p.value
    pw <- t.test(diff,mu=0,alternative='greater')$p.value
    results[[length(results)+1L]] <- data.frame(processing=krow$processing,contract=krow$contract,k=krow$k,
      reference=ref,mean_target_minus_ref=mean(diff),one_sided_t_p=pv,one_sided_worsening_p=pw,
      subjects_better=sum(diff<0),subjects_worse=sum(diff>0),subjects_tie=sum(diff==0))
  }
}
results <- do.call(rbind,results)
for(pr in unique(results$processing)) for(ref in unique(results$reference)) {
  i <- which(results$processing==pr & results$reference==ref)
  results$Holm_p_8tasks[i] <- p.adjust(results$one_sided_t_p[i],method='holm')
}
results$Holm_p_32_all_settings <- p.adjust(results$one_sided_t_p,method='holm')
results$Holm_p_worsening_32 <- p.adjust(results$one_sided_worsening_p,method='holm')
write.csv(results,file.path(out,'R544_R_independent_inference.csv'),row.names=FALSE)
locked <- read.csv(file.path(ind,'R544_fair_candidate_subject_bootstrap_Holm.csv'))
res <- merge(results,locked,by=c('processing','contract','k','reference'),suffixes=c('_R','_locked'))
stopifnot(nrow(res)==nrow(locked), max(abs(res$mean_target_minus_ref_R-res$mean_target_minus_ref_locked))<1e-7,
          max(abs(res$Holm_p_8tasks_R-res$Holm_p_8tasks_locked))<1e-5)
logit('PASS: all selection, mean risks and Holm-adjusted p-values match locked evidence')
stopifnot(nrow(policy)==720L,nrow(per)==15264L,nrow(results)==32L,
          nrow(do.call(rbind,checks))==10L,
          !anyNA(policy),!anyNA(per),!anyNA(results))
writeLines(c('PASS: FEATURE-LEVEL R544 REPLAY',
             'All 2,560 LDA refits, 720 fair candidate policy selections,',
             '15,264 subject-level risk records, archive exact-match checks and 32 inference comparisons passed.',
             paste('Completed:',format(Sys.time()))),
           file.path(out,'PASS_FINAL.txt'))
logit('[PASS] All 2,560 model refits, 720 policy choices, 15,264 subject risks and statistical checks completed')
logit('COMPLETED',format(Sys.time()))
