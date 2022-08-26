Search.setIndex({docnames:["index","pages/installation","pages/modules","pages/modules/data_preparator","pages/modules/distributions","pages/modules/experiment","pages/modules/filters","pages/modules/metrics","pages/modules/models","pages/modules/saver","pages/modules/scenarios","pages/modules/splitters","pages/modules/time","pages/overview","pages/spark","pages/useful","pages/useful_data/algorithm_selection","pages/useful_data/optuna"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst","pages/installation.rst","pages/modules.rst","pages/modules/data_preparator.rst","pages/modules/distributions.rst","pages/modules/experiment.rst","pages/modules/filters.rst","pages/modules/metrics.rst","pages/modules/models.rst","pages/modules/saver.rst","pages/modules/scenarios.rst","pages/modules/splitters.rst","pages/modules/time.rst","pages/overview.md","pages/spark.rst","pages/useful.rst","pages/useful_data/algorithm_selection.md","pages/useful_data/optuna.md"],objects:{"replay.data_preparator":[[3,0,1,"","DataPreparator"],[3,0,1,"","Indexer"]],"replay.data_preparator.DataPreparator":[[3,1,1,"","add_absent_log_cols"],[3,1,1,"","check_df"],[3,2,1,"","logger"],[3,1,1,"","read_as_spark_df"],[3,1,1,"","transform"]],"replay.data_preparator.Indexer":[[3,1,1,"","fit"],[3,1,1,"","inverse_transform"],[3,1,1,"","transform"]],"replay.distributions":[[4,3,1,"","item_distribution"],[4,3,1,"","plot_item_dist"],[4,3,1,"","plot_user_dist"]],"replay.experiment":[[5,0,1,"","Experiment"]],"replay.experiment.Experiment":[[5,1,1,"","__init__"],[5,1,1,"","add_result"],[5,1,1,"","compare"]],"replay.filters":[[6,3,1,"","filter_by_min_count"],[6,3,1,"","filter_out_low_ratings"],[6,3,1,"","take_num_days_of_global_hist"],[6,3,1,"","take_num_days_of_user_hist"],[6,3,1,"","take_num_user_interactions"],[6,3,1,"","take_time_period"]],"replay.metrics":[[7,0,1,"","Coverage"],[7,0,1,"","HitRate"],[7,0,1,"","MAP"],[7,0,1,"","MRR"],[7,0,1,"","NDCG"],[7,0,1,"","Precision"],[7,0,1,"","Recall"],[7,0,1,"","RocAuc"],[7,0,1,"","Surprisal"],[7,0,1,"","Unexpectedness"]],"replay.metrics.Coverage":[[7,1,1,"","__init__"]],"replay.metrics.Surprisal":[[7,1,1,"","__init__"]],"replay.metrics.Unexpectedness":[[7,1,1,"","__init__"]],"replay.metrics.base_metric":[[7,0,1,"","Metric"],[7,0,1,"","RecOnlyMetric"],[7,3,1,"","get_enriched_recommendations"]],"replay.metrics.base_metric.Metric":[[7,1,1,"","_get_metric_value_by_user"],[4,1,1,"","user_distribution"]],"replay.model_handler":[[9,3,1,"","load"],[9,3,1,"","save"]],"replay.models":[[8,0,1,"","ADMMSLIM"],[8,0,1,"","ALSWrap"],[8,0,1,"","AssociationRulesItemRec"],[8,0,1,"","ClusterRec"],[8,0,1,"","ImplicitWrap"],[8,0,1,"","ItemKNN"],[8,0,1,"","LightFMWrap"],[8,0,1,"","MultVAE"],[8,0,1,"","NeuroMF"],[8,0,1,"","PopRec"],[8,0,1,"","RandomRec"],[8,0,1,"","Recommender"],[8,0,1,"","SLIM"],[8,0,1,"","UCB"],[8,0,1,"","UserPopRec"],[8,0,1,"","Wilson"],[8,0,1,"","Word2VecRec"]],"replay.models.ADMMSLIM":[[8,1,1,"","__init__"]],"replay.models.ALSWrap":[[8,1,1,"","__init__"]],"replay.models.AssociationRulesItemRec":[[8,1,1,"","__init__"],[8,1,1,"","get_nearest_items"]],"replay.models.ClusterRec":[[8,1,1,"","__init__"]],"replay.models.ImplicitWrap":[[8,1,1,"","__init__"]],"replay.models.ItemKNN":[[8,1,1,"","__init__"]],"replay.models.LightFMWrap":[[8,1,1,"","__init__"]],"replay.models.MultVAE":[[8,1,1,"","__init__"]],"replay.models.NeuroMF":[[8,1,1,"","__init__"]],"replay.models.RandomRec":[[8,1,1,"","__init__"]],"replay.models.Recommender":[[8,1,1,"","fit"],[8,1,1,"","fit_predict"],[8,1,1,"","get_features"],[8,1,1,"","predict"],[8,1,1,"","predict_pairs"]],"replay.models.SLIM":[[8,1,1,"","__init__"]],"replay.models.UCB":[[8,1,1,"","__init__"]],"replay.models.Word2VecRec":[[8,1,1,"","__init__"]],"replay.models.base_rec.BaseRecommender":[[17,3,1,"","optimize"]],"replay.scenarios":[[10,0,1,"","Fallback"],[10,0,1,"","TwoStagesScenario"]],"replay.scenarios.Fallback":[[10,1,1,"","__init__"],[10,1,1,"","optimize"]],"replay.scenarios.TwoStagesScenario":[[10,1,1,"","__init__"],[10,1,1,"","fit"],[10,1,1,"","optimize"],[10,1,1,"","predict"]],"replay.session_handler":[[14,0,1,"","State"],[14,3,1,"","get_spark_session"]],"replay.splitters.base_splitter.Splitter":[[11,3,1,"","split"]],"replay.splitters.log_splitter":[[11,0,1,"","ColdUserRandomSplitter"],[11,0,1,"","DateSplitter"],[11,0,1,"","NewUsersSplitter"],[11,0,1,"","RandomSplitter"]],"replay.splitters.log_splitter.ColdUserRandomSplitter":[[11,1,1,"","__init__"]],"replay.splitters.log_splitter.DateSplitter":[[11,1,1,"","__init__"]],"replay.splitters.log_splitter.NewUsersSplitter":[[11,1,1,"","__init__"]],"replay.splitters.log_splitter.RandomSplitter":[[11,1,1,"","__init__"]],"replay.splitters.user_log_splitter":[[11,0,1,"","UserSplitter"],[11,3,1,"","k_folds"]],"replay.splitters.user_log_splitter.UserSplitter":[[11,1,1,"","__init__"]],"replay.time":[[12,3,1,"","get_item_recency"],[12,3,1,"","smoothe_time"]],replay:[[6,4,0,"-","filters"],[7,4,0,"-","metrics"],[8,4,0,"-","models"],[10,4,0,"-","scenarios"],[11,4,0,"-","splitters"]]},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","property","Python property"],"3":["py","function","Python function"],"4":["py","module","Python module"]},objtypes:{"0":"py:class","1":"py:method","2":"py:property","3":"py:function","4":"py:module"},terms:{"0":[3,5,6,7,8,10,11,12,16,17],"00":[3,6,12],"000000":[5,8],"005":16,"009":16,"01":[3,6,8],"013":16,"017":16,"019667":8,"02":6,"025":8,"026":16,"028":16,"03":12,"031":16,"033":16,"034":16,"04":[6,16],"045":16,"05":[6,8],"056":16,"06":17,"068":16,"069":16,"072":16,"084":16,"09":5,"092":16,"093":16,"1":[3,4,5,6,7,8,10,11,12,17],"10":[4,6,8,10,11,12,16,17],"100":[8,10],"1000":8,"111":16,"118":16,"12":[8,12,16],"123":[10,16],"124":16,"125":8,"128":[8,16],"13":[5,16],"133":16,"138":16,"14":6,"142":16,"145":16,"15":6,"151":16,"157":16,"159":16,"16":16,"163":16,"166":16,"167":16,"168":16,"17":16,"172":16,"181":16,"183":16,"19":12,"193":16,"195":16,"1e":17,"1k":7,"1m":0,"2":[3,5,6,7,8,11,12,14,17],"20":[8,11,12],"200":8,"2019":3,"2020":6,"206549":8,"2099":[3,12],"22":12,"23":6,"232":16,"235":16,"237":16,"238":16,"24":16,"244":16,"25":[7,8,12,16],"255":16,"256":16,"257":16,"26":[12,16],"261":16,"262":16,"263":16,"265":16,"266":16,"27":12,"275":8,"2_":8,"2_2":8,"2_f":8,"2i":8,"2n":8,"3":[1,3,5,6,7,8,11,12,16],"30":[11,12],"304":16,"318":16,"33":5,"330329915368074":12,"333333":5,"3333333333333333":8,"350":16,"36":5,"367":16,"382":16,"386853":5,"39":16,"396":16,"3_0":5,"3_median":5,"4":[5,6,7,8,11,12],"40":11,"409":16,"412":16,"414":16,"419":16,"42":11,"431":16,"442027":8,"456":16,"469279":5,"4n":8,"5":[3,5,6,7,8,11,12,17],"50":[8,10],"5000":8,"515":16,"519667":8,"530721":5,"537":16,"555":8,"59":6,"591":16,"6":[7,8,11,16],"600":8,"613147":5,"625":8,"627":16,"6390430306850825":12,"64":16,"645":16,"647":16,"654":16,"654567945027101":12,"655":16,"6632341020947187":12,"666667":[5,8],"669":16,"67":[7,11],"684":16,"686":16,"691":16,"6940913454809814":12,"699":16,"7":[1,7,8,16],"70":14,"704":16,"7203662792445817":12,"737":16,"75":[7,8],"77":16,"777":8,"7994016704292545":12,"8":[7,8],"80083":11,"827":16,"8312378961427874":12,"833":16,"8506671609508554":12,"86":16,"8605514372443298":12,"8645248117312496":12,"8666666666666667":12,"876":16,"8833333333333333":12,"890898718140339":12,"8916666666666666":12,"898":16,"9":12,"91":5,"9117224885582166":12,"9166666666666666":12,"9333333333333333":12,"95":5,"9548416039104165":12,"95_conf_interv":5,"961":16,"964":16,"9666666666666667":12,"977":16,"\u043e\u0431\u0435\u0440\u0442\u043a\u0430":8,"abstract":7,"case":[7,8,17],"class":[3,5,7,8,10,11,13,14],"default":[3,7,8,10,11,13,14,17],"do":7,"final":[7,8],"float":[3,5,7,8,11,12],"function":[1,3,7,8,9,12,13,14],"import":[3,5,6,7,8,11,12,13,14,17],"int":[4,5,6,7,8,10,11,14,17],"new":[0,7,8,10,11,16,17],"null":3,"return":[3,4,5,6,7,8,9,10,11,12,14,17],"static":[3,7],"true":[5,6,7,8,10,11,17],"try":[8,10,17],"while":16,A:[8,17],And:3,But:12,By:[7,8,11,13,16],For:[1,17],If:[1,3,4,6,8,10,13,17],In:[7,8,17],It:[1,7,8,13],No:1,Or:11,That:7,The:[6,7,8,16],There:[12,13,14],These:12,To:[1,7,8,13],Will:16,With:8,_1:8,_2:8,_:[7,8],__init__:[5,7,8,10,11],_f:8,_fit:1,_fit_wrap:1,_get_enriched_recommend:7,_get_metric_value_by_us:7,_pair:1,_predict:1,_predict_by_us:1,_predict_pairs_inn:1,_predict_selected_pair:1,_search_spac:17,_wrap:1,a_j:8,abl:16,about:[3,13],abracadabra:8,absenc:1,absent:3,access:14,account:[7,12],achiv:8,across:[5,7],action:[8,10,11],actual:[7,12],ad:[0,8],add:[1,3,7,8,10],add_absent_log_col:3,add_cold:8,add_result:5,adjust:8,admm:[0,2,16],admmslim:8,after:12,afterward:12,ag:12,aggreg:7,ai:1,al:[8,10,16],algorithm:[8,16],all:[4,7,8,10,11,13,14,16,17],alloc:14,allow:[8,11],alpha:[7,8],alreadi:7,also:[1,7,8,13,14],alswrap:[8,10],altern:[0,2,7,16],alternatingleastsquar:8,among:[3,7,8],amount:[8,11],an:[1,8,12],ani:[5,8,10,14,16,17],anneal:8,ap:7,aposteriori:8,appear:[6,7,8],appli:[6,8,12],approxim:8,apt:1,ar:[3,5,6,7,8,10,11,12,13,14,16,17],arbitrari:3,architectur:1,area:7,arg:[7,17],argument:[3,12],around:12,arrai:[8,10],arraytyp:8,ascend:8,assign:11,associ:[0,2,16],associationrulesitemrec:[1,8],atm:11,attribut:[5,14,17],auc:[0,2],augment:10,autoencod:8,automat:[11,13,14],avail:[3,5,7,11,12,14],averag:[4,7],await:3,b:8,back:[3,13],bandit:8,base:[7,8,10,11,12,16],base_metr:7,base_rec:[8,17],base_splitt:11,baselin:[5,7],baserecommend:[8,9,10,17],basetorchrec:1,basic:[0,8,14],becaus:[7,13],befor:8,begin:12,behav:10,best:[8,10,17],beta:[8,17],better:[7,8],between:[6,8],bigger:[7,8],biggest:[7,13],bin:4,binari:[7,8,16],binomi:8,bm25:8,bool:[5,6,8,10,11,17],border:[8,17],both:[3,8,16],bound:[7,8],budget:[8,10,17],build:[1,8],built:8,c:[1,8,12],caclul:8,calc_conf_interv:5,calc_median:5,calcul:[4,5,6,7,8,12,16],call:[8,17],can:[1,3,4,7,8,9,10,11,12,13,14,16,17],candid:[8,10],cat_1:[8,17],cat_2:[8,17],cat_3:[8,17],cat_param:[8,17],categor:[8,10,17],cd:1,cdot:8,center:3,chain:1,chang:7,characterist:7,check:3,check_df:3,choos:[0,8,15],chosen:11,classic:8,classif:7,clone:1,closer:8,closest:4,cluster:[0,2,16],clusterrec:[1,8],co:8,coeffici:8,coefici:8,col:4,cold:[7,8,10,11,16],colduserrandomsplitt:[0,2],collabor:[8,16],colour:4,column:[3,4,6,7,8,10,12,13],columns_map:3,com:1,combin:8,commonli:8,compar:[0,2],comparison:[0,15],compil:1,complet:[7,16],compon:1,conf_interv:7,confid:[5,8],confidence_gain:8,config:10,configur:1,consid:8,consist:11,constant:12,consum:7,contain:[3,7,8,10],content:16,continu:17,convers:8,convert2spark:[3,6,8,11,13],convert:[3,8,13,16],core:1,correct_log:3,correctli:7,correspond:[11,12],cosin:8,could:8,coun:6,count:[6,7,8,14,16],count_negative_sampl:8,coverag:[0,2,5,16],cpu:[8,14],creat:[3,8,10,13,14],creation:[10,14],criteria:6,criterion:[8,10,17],csv:3,ctr_i:8,cuda:14,cumul:7,current:[1,7],curv:7,custom:[0,2,10,13],custom_features_processor:10,cut:[4,5,7],cython:1,d:[8,12],dai:[6,12],data:[0,2,4,5,6,7,8,10,11,15,17],data_fram:[6,8,11],data_prepar:3,datafram:[3,4,5,6,7,8,10,11,12,13,17],dataprepar:3,dataset:[6,10,11],datatyp:3,date:[3,6,11,13],date_col:6,date_column:6,date_format:3,datesplitt:[0,2],datetim:[6,11,13],dcg:7,dd:[6,11],debug:14,decai:12,decod:8,decreas:10,default_relev:3,default_t:3,defin:[3,8],denois:8,denomin:8,densiti:8,depend:[1,7,12,16],depth:[4,5,7],destin:9,determin:11,dev:1,develop:0,devic:14,df:[3,8,12],dfrac:8,dict:[3,5,8,10,17],dictionari:[3,5,7,8,10,17],differ:[3,5,8,16,17],dimens:[8,13],dimz:8,direct:8,directori:1,discount:7,disk:9,distanc:8,distinct:7,distribut:[0,2],diverg:8,divers:7,divid:[7,10],document:8,doe:[8,12],doesn:1,don:[13,16],dot:8,doubletyp:3,dp:3,drastic:11,drop:[8,11],drop_cold_item:11,drop_cold_us:11,drop_zero_rel_in_test:11,dropout:8,duration_dai:6,dure:[8,10],e:[1,6,8,16],each:[6,7,8,10,11,12],effici:8,either:[3,6,8,16,17],elasticnet:8,embed:8,embedding_gmf_dim:8,embedding_mlp_dim:8,empti:[3,10],encod:[8,13],end_dat:6,entiti:[6,13],entri:[6,11],epoch:8,equal:[8,12],error:1,essenti:1,estim:[8,13],ether:3,even:11,everi:[7,12],ex:5,exact:11,exampl:[1,3,4,5,8,10,11,12,17],exp:12,experi:5,explicit:[8,16],explor:[8,17],exploration_coef:8,exponenti:12,extens:1,extra:[3,7,8],f0:3,f1:3,factor:[0,2,16],fall:12,fallback:[0,2],fallback_model:10,fals:[5,6,8,10,11,17],feat:16,featur:[3,8,10,16,17],feature1:3,feature2:3,feedback:[8,16],fewer:8,file:[1,3,9,10],fill:10,filted:6,filter:[0,2,8],filter_by_min_count:6,filter_out_low_r:6,filter_seen_item:[8,10],find:8,first:[6,7,10,17],first_level:10,first_level_model:10,first_level_train:10,first_stage_model:10,first_stage_train:10,fit:[1,3,8,9,10],fit_pred_tim:16,fit_predict:8,fix:[1,8],flag:[5,8,10,11],fold:11,folder:9,follow:[1,7],form:[3,8],format:[0,3,5,6],format_typ:3,formul:8,frac:[7,8],fraction:[7,8,11],framework:1,frequenc:8,from:[1,3,5,6,7,8,9,10,11,14,17],g:[1,6,8,16],gain:7,gap:13,gaussian:8,gb:14,ge:8,gener:[1,3,8,10],geqslant:8,get:[0,1,3,4,6,7,8,10,14,16],get_enriched_recommend:7,get_featur:8,get_item_rec:12,get_nearest_item:8,get_spark_sess:[7,8,11,14],getlogg:14,git:1,github:1,give:16,given:[7,8],gmf:8,goe:16,gpu:8,grater:12,greater:8,grid:10,ground:7,ground_truth:[4,7],group:[6,8],group_bi:6,gt_:7,h:1,ha:[3,7,11,12],half:[5,12],handl:3,happen:[7,12],hat:8,have:[1,7,8,10,13,16,17],header:1,help:6,helper:14,here:[4,7,14],hh:6,hidden:8,hidden_dim:8,hidden_mlp_dim:8,high:[8,10,17],histor:[4,6,7,8,10],histori:[4,11,16],historybasedfeaturesprocessor:10,hitrat:[0,2,16],holdout:7,hood:8,hot:10,how:[0,6,7,8,15],hybrid:16,hyper:[0,15],i1:6,i2:6,i3:6,i:[7,8],i_u:8,id:[0,3,8],idcg:7,identif:13,idx:[3,13],ignor:[6,7,8],ignore_index:8,ij:[7,8],implement:[7,8],implicit:[0,1,2,16],implicit_pref:8,implicitwrap:[1,8],improv:8,includ:8,increas:8,independ:8,index:[0,3,13],indic:[1,7],infer:[0,2,10],info:[0,8,13,14,16],inform:[7,16],inherit:[1,7],initi:[3,5,7,8,10,17],input:[0,3,8,10,11,15],insid:11,instal:0,instead:8,integ:[3,13],interact:[3,6,7,8,10,12,16],interfac:[0,1,2,10],intern:[3,13],interpret:7,interv:[5,8],invers:[7,8],inverse_transform:3,item:[0,2,3,6,7,10,11,12,13,16,17],item_cat_features_list:10,item_col:[3,6],item_dist:4,item_distribut:4,item_featur:[8,10,17],item_id:[3,7,8,10,11],item_idx:[3,5,6,7,8,10,11,12,13],item_popular:8,item_test_s:11,itemknn:[8,16],iter:[4,5,7,8,10],its:8,j:7,jj:8,json:3,just:17,k:[0,2,4,7,10,16,17],k_fold:[0,2],keep:[8,10,17],kei:[3,5,8,10,17],kind:12,kl:8,kwarg:[3,7],l1:8,l2:8,l2_reg:8,l:8,lab:1,label:13,lambda:8,lambda_1:8,lambda_2:8,lambda_:[8,17],last:[6,8,11,12],latent:8,latent_dim:8,later:9,le:7,learn:8,learning_r:8,least:[0,2,7,16],leav:6,left:[3,8],length:[4,6,7,8,10,17],less:[6,8],level:[10,14,16],librari:[3,8,14],lift:8,lightautoml:10,lightfm:[0,1,2,16],lightfmwrap:[1,8],like:[4,8,10],likelihood:8,limit:12,limits_:8,linear:[8,12],list:[4,5,7,8,10,17],ln:8,load:[3,9],local:8,lock:1,log:[0,3,4,5,6,7,8,10,11,12,16],log_2:7,log_pd:6,log_sp:6,log_splitt:11,logger:[3,14],loguniform:17,look:14,loss:[8,16],low:[8,10,17],lower:[7,8],lr:8,machin:1,made:8,magma:4,main:10,main_model:10,major:12,mani:[6,7,8],map:[0,2,3,5,16],mathbb:[7,8],mathcal:8,matric:13,matrix:[0,2,16],max:[7,8],max_:7,max_it:8,maxim:8,maximum:8,mean:[4,7,8],measur:[7,8],median:[5,7],memori:14,merg:7,method:[4,7,8,11],metric:[0,2,4,5,8,10,16,17],min_count:8,min_item_count:8,min_pair_count:8,minim:[6,8,12],minimum:8,miss:10,mlp:8,mm:[6,11],mode:8,model:[0,2,4,5,7,9,10,13,15],model_handl:9,moder:11,modif:8,modifi:[8,12],modul:[0,8,12,14],more:[7,8,12,16],most:[7,8,10,11],movielen:[0,4],mrr:[0,2,16],mu:8,mu_:8,mu_i:8,mult:[0,2,16],multipl:7,multipli:8,multva:[8,16],must:[8,12],n:[7,8],n_fold:11,n_i:8,n_iu:8,n_u:8,name:[3,5,6,8,10,13,14,17],natur:3,ncf:8,ndcg:[0,2,5,8,10,16,17],nearest:[0,2,16],necessari:1,need:[1,7,16],neg:[8,10],negatives_typ:10,neighbor:8,neighbour:[0,2,16],neighbour_item_id:8,neighbourrec:1,net:8,network:8,neumf:8,neural:[0,2,16],neuromf:[8,16],new_lr:8,new_studi:[8,10,17],newuserssplitt:[0,2],no_compon:8,non:8,none:[3,5,6,8,10,11,14,17],normal:[7,8],num_candid:10,num_clust:8,num_entri:6,num_interact:6,num_neg:10,num_neighbour:8,number:[3,4,6,7,8,10,11,12,14,16,17],numer:[3,13],nuniqu:11,object:[4,7,8,10,12,17],observ:8,occurr:[6,8],off:[4,5,7],old:[8,12],one:[3,7,8,10,11,12,13,16,17],onli:[3,6,7,11,12,16,17],onlin:8,oper:7,optim:[0,8,10,15],option:[3,5,6,8,10,11,14,17],optuna:[1,8,10,17],order:[6,7],orderbi:[6,12],origin:3,other:[0,1,2,12,14,16,17],our:16,out:8,output:4,own:14,p:8,p_:8,p_d:8,packag:1,page:[0,8,13,17],pair:[7,8,10],palett:4,panda:[3,4,5,6,7,8,11,12,13],pandas_df:5,param:[3,8,10,17],param_bord:[8,10,17],paramet:[0,3,4,5,6,7,8,9,10,11,12,14,15,17],parquet:3,part:[6,11],partial:8,partit:14,pass:[3,8,10,14],path:[3,9,10],patienc:8,pd:[3,4,5,6,7,8,11,12],per:[4,6,10,11],percentag:[5,7],perform:[1,3,7],person:[8,16],phi:8,pip:1,plai:16,pleas:[8,17],plot:4,plot_item_dist:4,plot_user_dist:4,pm:8,poetri:1,point:[8,10,17],pop_rec:10,poprec:[8,10,16],popular:[0,2,4,16],popular_bas:8,posit:[7,8,10],possibl:[3,7,8,10,17],power:12,precis:[0,2,5,10],pred:[5,7],pred_i:8,predict:[1,4,5,7,8,10,16],predict_pair:[1,8],prefer:13,prepar:[0,2],preper:8,preprocess:3,present:[3,6,7,8,16],previou:[8,10,17],probabl:8,problem:8,process:[3,8,16],processor:10,produc:8,properti:3,propos:8,propto:8,provid:[8,12,14],put:[11,14],pypandoc:1,pyproject:1,pyspark:[8,13],python3:1,python:[1,8],pytorch:[8,14],q_:8,qualiti:8,quantil:[5,8],quickli:12,r:8,r_:7,ram:14,random:[0,2,10,11,16],random_pop:8,random_st:8,randomli:11,randomrec:[1,8,16],randomsplitt:[0,2],rang:[8,10,17],rank:[7,8],rank_i:7,rare:[7,8],rate:[4,7,8,13,16],rating_column:6,raw:[3,13],re:8,reach:[8,12],read:3,read_as_spark_df:3,reader_kwarg:3,rec:[1,5,7,8],rec_count:4,recal:[0,2],receiv:7,recent:[8,11],reciproc:7,recognit:8,recommend:[0,1,2,4,5,7,9,10,15,17],reconlymetr:7,record:[5,6,7,11],reduc:[8,11,12],reducelronplateau:8,refer:13,regard:13,regress:8,regular:8,rel:[3,6],rel_i:7,relev:[3,5,6,7,8,10,11,12,13],remain:6,remov:[6,8,10,11],renam:3,replai:[1,2,3,4,5,6,7,9,10,11,12,13,14,16,17],repositori:1,represent:13,request:1,requir:[0,1,3,7,8,15],reset_index:8,resolv:1,restor:9,result:[0,2,4,7,8,12,17],retrain:16,retriev:11,reweight:8,right:[8,17],rightarrow:8,roc:[0,2],rocauc:7,round:7,row:[3,8],rtype:3,rule:[0,2,16],run:[1,5],s:[7,8,10,11],same:[4,8,10,14,16],sampl:8,save:[8,9],sb:1,scenario:[0,2,16],scheme:4,score:7,seaborn:4,search:[0,8,10,17],searchabl:17,second:[10,16],second_level_train:10,second_model_config_path:10,second_model_param:10,see:[8,17],seed:[8,10,11],seen:[8,10,13],select:6,self:[7,11,17],sentenc:8,separ:[8,11],seri:10,serial:[0,2],session:[0,8,13],session_col:8,session_handl:[7,8,11,14],set:[0,7,8,11,17],setlevel:14,should:[3,6,7,8,13,16],show:[3,4,5,6,7,8,11,12],shown:7,shrink:8,shuffl:11,shuffle_partit:14,si:7,sigma:8,sigma_:8,sigma_i:8,sim:8,similar:[8,11],simpl:8,simultan:[6,7],singl:7,size:[8,11],skip:10,slim:[0,2,16,17],slowli:12,small:11,smooth:[0,2,4,8],smoothe_tim:12,so:[3,7,8,11,12],solut:8,solv:8,some:[6,7,8,12,16],someth:4,sort:6,sort_valu:8,spark:[0,3,6,7,8,11,13],spark_memori:14,sparksess:14,spars:[8,13],specif:8,specifi:[3,7,10,11,17],split:[8,10,11],splitter:[0,2,10],sqrt:8,squar:[0,2,16],ss:6,stage:[0,2,7,16],standard:3,start:[0,8,10,17],start_dat:6,state:[7,8,11,14],statist:10,step:8,step_siz:8,stop:17,store:[5,9,13,14,17],str:[3,4,5,6,8,9,10,11,12,17],strategi:[8,10,11],string:11,studi:[8,10,17],sudo:1,sum:8,sum_:[7,8],surpris:[0,2,5,16],system:[7,8],t:[1,7,8,13,16],tabl:[3,5],tabularautoml:10,take:[6,7,8,10,11,12],take_num_days_of_global_hist:6,take_num_days_of_user_hist:6,take_num_user_interact:6,take_time_period:6,term:8,test:[4,5,7,8,10,11,17],test_siz:11,test_start:11,textit:7,tf_idf:8,than:[6,7,8,12],thei:[7,8,16],them:[3,8,9,10,17],theta:8,thi:[1,3,4,5,7,8,12,13,14,17],think:13,three:12,threshold:10,thu:8,ti:7,time:[0,2,6,8],timestamp:[0,3,6,7,8,10,11,12],timestamptyp:3,titl:4,tj:7,to_datetim:6,token:8,toml:1,too:14,top:[3,7,8,10],topanda:[6,8,11],total:8,toward:8,traceback:8,train:[1,8,9,10,11,16,17],train_splitt:10,transform:[3,12,16],treat:[7,8],trial:17,tripl:14,trivial:8,troubl:1,troubleshoot:0,truncat:7,truth:7,ts:3,tupl:[8,10,11],two:[0,2,16],twostagesscenario:10,type:[3,4,5,6,7,8,9,10,11,12,14,16,17],typic:1,u1:6,u2:6,u3:6,u:[1,8],u_j:7,ubuntu:1,ucb:[0,2,16],unari:16,under:[7,8],unexpect:7,unexpected:[0,2],uniform:[8,16],uniformli:8,union:[3,4,5,6,7,8,10,11,12],uniqu:8,univers:14,unix:[1,11],unseen:16,updat:[1,12],upgrad:1,upper:8,us:[0,1,3,4,6,7,8,10,12,13,14,16,17],use_first_level_models_feat:10,use_generated_featur:10,use_idf:8,use_relev:8,user1:3,user2:3,user:[0,2,3,5,6,7,10,11,13,16,17],user_cat_features_list:10,user_col:[3,6],user_dist:4,user_distribut:4,user_featur:[8,10,17],user_id:[3,7,8,10,11],user_idx:[3,5,6,7,8,10,11,13],user_item_popular:8,user_log_splitt:[10,11],user_test_s:11,userpoprec:8,usersplitt:[0,2,10],usual:[1,8],util:[6,8,11,13],vae:[0,2,16],val:17,valu:[3,4,5,6,7,8,10,12,17],valueerror:8,varepsilon:8,variabl:8,variant:[8,11],variat:8,vector:8,version:[1,7],via:14,view:16,vocabulari:8,w:[8,16],w_:8,w_j:8,wa:7,wai:[8,12,17],want:[8,13,17],warn:3,warp:8,we:[1,3,7,8,17],weigh:12,weight:[8,12,13],were:7,what:16,wheel:1,when:[8,12],where:[7,8,9,10,17],whether:7,which:[6,7,8,10,12],who:8,whole:7,widehat:8,wilson:[0,2,16],wilsonscor:8,window:[4,8],window_s:8,without:[8,13,16],word2vec:[0,2,16],word2vecrec:[1,8],word:8,wrap:16,wrapper:[0,2],wrong:[3,10],x:8,x_i:8,yield:11,you:[1,3,4,7,8,9,10,11,13,14,17],your:[3,7,13,14],yourself:[3,17],yyyi:[6,11],z:8,z_:8,zero:13},titles:["Welcome to RePlay\u2019s documentation!","Installation","Modules","Data Preparation","Distributions","Compare Results","Filters","Metrics","Models","Serializer","Scenarios","Splitters","Time Smoothing","Get Started","Settings","Useful Info","How to choose a recommender","How to optimize a model"],titleterms:{"1m":16,"new":1,ad:1,admm:8,altern:8,associ:8,auc:7,basic:1,choos:16,cluster:8,colduserrandomsplitt:11,compar:5,comparison:16,content:[0,2,15],coverag:7,custom:7,data:[3,13,16],datesplitt:11,develop:1,distribut:[4,8],document:0,factor:8,fallback:10,filter:6,format:13,get:13,hitrat:7,how:[16,17],id:13,implicit:8,indic:0,infer:8,info:15,input:16,instal:1,interfac:8,item:[4,8],k:8,k_fold:11,least:8,lightfm:8,log:14,map:7,matrix:8,metric:7,model:[1,8,16,17],modul:2,movielen:16,mrr:7,mult:8,ndcg:7,nearest:8,neighbour:8,neural:8,newuserssplitt:11,optim:17,other:8,popular:8,precis:7,prepar:3,random:8,randomsplitt:11,recal:7,recommend:[8,16],replai:[0,8],requir:[13,16],result:5,roc:7,rule:8,s:0,scenario:10,serial:9,session:14,set:14,slim:8,smooth:12,spark:14,splitter:11,squar:8,stage:10,start:13,surpris:7,tabl:0,time:12,timestamp:13,troubleshoot:1,two:10,ucb:8,unexpected:7,us:15,user:[4,8],usersplitt:11,vae:8,welcom:0,wilson:8,word2vec:8,wrapper:8}})