###Table of content
1. [General info](#general-info)

2. [Important files](#important-files)

3. [Important directories](#important-directories)

4. [Short list of important supplementary commands](#sup-commands)

5. [How to deploy DAGs to an Airflow instance](#deploy-dags)

6. [How to run experiments with Airflow](#experiments)

7. [Workflow structure update and backfilling](#backfilling)

8. [External and supplementary services to support workload executing](#external-services)

### <a name="general-info"></a>General information.
A typical scenario of how it works is the following:
1.  Create a DAG object by calling a builder function with appropriate parameters in **dag_two_stage_scenarios.py**.
    ```python
    ml1m_first_level_dag_submit = build_fit_predict_first_level_models_dag(
            dag_id="ml1m_first_level_dag_submit",
            mlflow_exp_id="111",
            model_params_map=_get_models_params("als"),
            dataset=DATASETS["ml1m"]
    )
    ```
    Here, **ml1m_first_level_dag_submit** is a dag object with id *ml1m_first_level_dag_submit* 
    (depicted on the web ui). **build_fit_predict_first_level_models_dag** is a builder function 
    that creates that the DAG object, while *mlflow_exp_id*, *model_params_map*, *dataset* are params used 
    to parametrize the builder function to build a DAG specifically for ml1m dataset.
    

2.  Deploy updated **dag_two_stage_scenarios.py** to an Airflow instance by means of uploading this file 
    and its dependencies (other dag_*.py files) to a folder on a remote server.
    

3. Build or update docker images if necessary. Push these images into the common cluster repository (node2.bdcl:5000). 

   
4.  Go to the Web UI, ensure it has uncovered the new version (may take a few moments) and run the DAG from the Web UI


5.  If some crashes and fails appear, than check the log, check [the history server](#external-services), 
    fix the code/env/params, redeploy if necessary 
    and clear status of the failed task (in the task context menu). 
    Airflow will automatically reschedule a new instance of the task. 
    

**Important Note.** If a workflow run already exists (e.g. simply speaking some failed, running or finished tasks 
are associated with the workflow run) and one updates the structure of this workflow, 
newly added tasks **won't be scheduled**. One needs either to create a new workflow run in the Web UI 
or backfill the new tasks (check section [Workflow structure update and backfilling](#backfilling)). 


**Important Note.** Business logic of the tasks should be written idempotent. In other words, 
if there are already results of previous execution of the task, the task should not to start new computations 
and should end successfully unless **forced reexecution** is explicitly stated (and supported by the task logic). 
The user, if he wants to calculate tasks anew, should either delete existing results before tasks starting 
or explicitly state forced reexecution (via some parameter of DAG or task). 

### <a name="important-files"></a>Important files.

The files structure related to running airflow DAGs 
1. **experiments/dag_two_stage_scenarios.py** - contains definitions of tasks (wrapper functions that create task 
   objects) and DAGs in the form of python functions that build DAG objects. 
   

2. **experiments/dag_entities.py** - an utility file that provides functionality to build workflows. In particular, 
   the template of spark-submit command used by SparkSubmit operator to run RePlay workload on YARN.    


3. **experiments/dag_utils.py** - contains functions with the business logic of data processing / models training  


4. **experiments/bin/replayctl** - a command line executable that provides commands for all main actions related 
   to deploying workload, providing access to Airflow WebUI, to build docker images and etc.


5. **experiments/docker** - contains docker files to build an image of airflow executor for Kubernetes.

### <a name="important-directories"></a>Important directories

1. **Python environments**: /mnt/ess_storage/DN_1/storage/SLAMA/python_envs

2. **General project directory**: /mnt/ess_storage/DN_1/storage/SLAMA/kaggle_used_cars_dataset

### <a name="sup-commands"></a>Short list of important supplementary commands
1. To show airflow config, pod template, etc. 

    ```shell
        kubectl -n airflow get configmap/airflow-airflow-config -o yaml
    ```
    
    Specifically, the pod template is stored in the section **pod_template_file.yaml**.


2. To list available commands for the project management:

    ```shell
        ./experiments/bin/replayctl help
    ```


3.  To backfill new tasks freshly added to a workflow: 
    ```shell
        airflow dags backfill -v --start-date=2022-12-07 -t second_level_lama_single_lgb -i 2stage_ml1m_alswrap
    ```
    
    Check out [the corresponding section](#backfilling) for more info. 


4.  To access logs of drivers and executors running or finished/failed on YARN:
    ```shell
        # execute it only once
        alias kyarn="kubectl -n spark-lama-exps exec -it deployment/yarn-resourcemanager -- yarn logs"
        
        # find out ids of YARN containers (don't miss it with docker container - it is completely different staff)
        kyarn -applicationId application_1665511485031_0867 -show_application_log_info
    
        # checking out all log files of a particular container. 
        kyarn -applicationId application_1665511485031_0867 -containerId container_e46_1665511485031_0867_01_000001
        
        # checking out only stdout lof of a particular container.
        kyarn -applicationId application_1665511485031_0867 -containerId container_e46_1665511485031_0867_01_000001  -log_files=stdout 
    ```
    
    **Important Notes:** An application Id can be found in logs of airflow tasks (from Airflow Web UI) 
    or through [YARN Web UI](#external-services)

    **Important Note:** A YARN container that hosts driver of a Spark app has id '1'.

    **Important Note:** Be aware of stdout and stderr log files may contain different info 
    and thus one may have to check both of them to find out the reason of a crush.


5.  To kill a Spark application running on YARN:  
    ```shell
        # execute it only once
        alias kykill="kubectl -n spark-lama-exps exec -it deployment/yarn-resourcemanager -- yarn application -kill"
        
        # kills a distributed app including its driver and all executors
        kykill -appId application_1665511485031_0867 
    ```

### <a name="deploy-dags"></a>How to deploy DAGs to an Airflow instance
    
1. To update list of availables workflows, to change workflow structures or 
   to update source codes of functions in the workflows run the command:
   ```shell
      ./experiments/bin/replayctl sync-dags
    ```
    
   It requires presence of **rsync** binary installed on the workstation and ssh access to blade servers. 
   The command copies dag_*.py files from 'experiments' folder to a folder on ESS. 
   Path of the remote folder: */mnt/ess_storage/DN_1/storage/Airflow/dags*. 
   This folder is shared with various airflow services responsible for workflow execution, for instance, 
   DAG scheduler and Web UI services.
   
   On should remember that it is not possible to alter a workflow directly from Web UI, but only to monitor 
   and affect its execution, including launching a workflow, pausing it, restarting failed tasks.
   
   In order to deploy test/development dag scripts assigned to a specific user, 
   you must declare the `USERNAME_ON_CLUSTER` environment variable.
   The `USERNAME_ON_CLUSTER` user must have ssh access to the dag scripts directory. 
   The most convenient way is to write `export USERNAME_ON_CLUSTER=username` in the ~/.bashrc file.
   Next, you need to create the dag_*_username.py files in the `experiments` directory
   and after that the `./experiments/bin/replayctl sync-dags` command will deploy your dag scripts to the airflow directory.
   Your dag scripts will be copied with the `_username` suffix and will not conflict with other users' files.

2.  To execute a task Airflow is configured to use Kubernetes executor that runs tasks as separate pods 
    inside Airflow namespace on Kubernetes. 
    This is also true for tasks that submit Spark jobs to a standalone YARN cluster. In such a case, 
    the pod will execute spark-submit script that starts a job on YARN and 
    is constantly reporting status of the job until it is finished or failed.
    
    Each pod is created from the following template which is a content of pod_template_file.yaml 
    as it was mentioned earlier. 
    The current version of the pod template 
    (be aware that the actual pod template may differ one better check it for himself using the command above):
    
    <a name="pod-template"></a>
    ```yaml
     apiVersion: v1
     kind: Pod
     metadata:
       name: dummy-name
     spec:
       containers:
         - env:
             - name: AIRFLOW__CORE__EXECUTOR
               value: LocalExecutor
             # Hard Coded Airflow Envs
             - name: AIRFLOW__CORE__FERNET_KEY
               valueFrom:
                 secretKeyRef:
                   name: airflow-fernet-key
                   key: fernet-key
             - name: AIRFLOW__DATABASE__SQL_ALCHEMY_CONN
               valueFrom:
                 secretKeyRef:
                   name: airflow-airflow-metadata
                   key: connection
             - name: AIRFLOW_CONN_AIRFLOW_DB
               valueFrom:
                 secretKeyRef:
                   name: airflow-airflow-metadata
                   key: connection
           image: node2.bdcl:5000/airflow-worker:latest
           imagePullPolicy: Always
           name: base
           resources:
             requests:
               memory: "30Gi"
               cpu: "6"
             limits:
               memory: "30Gi"
               cpu: "6"
           volumeMounts:
             - mountPath: /opt/airflow/airflow.cfg
               name: airflow-config
               readOnly: true
               subPath: airflow.cfg
             - name: data
               mountPath: /opt/spark_data/
             - name: mlflow-artifacts
               mountPath: /mlflow_data/artifacts
             - name: logs
               mountPath: "/opt/airflow/logs"
             - name: dags
               mountPath: /opt/airflow/dags
             - name: ephemeral
               mountPath: "/tmp"
       restartPolicy: Never
       securityContext:
         runAsUser: 50000
         fsGroup: 50000
       serviceAccountName: "airflow-worker"
       volumes:
         - configMap:
             name: airflow-airflow-config
           name: airflow-config
         - name: data
           persistentVolumeClaim:
             claimName: spark-lama-data
             readOnly: false
         - name: mlflow-artifacts
           persistentVolumeClaim:
             claimName: mlflow-artifacts
         - name: logs
           persistentVolumeClaim:
             claimName: airflow-logs
         - name: dags
           persistentVolumeClaim:
             claimName: airflow-dags
         - name: ephemeral
           emptyDir: {}
    ```
    
    To alter the pod's configuration one may apply ad-hoc patch 
    in DAGs file **dag_two_stage_scenarios.py** the following way: 
    ```python
    
    extra_big_executor_config = {
        "pod_override": k8s.V1Pod(
            spec=k8s.V1PodSpec(
                containers=[
                    k8s.V1Container(
                        name="base",
                        resources=k8s.V1ResourceRequirements(
                            requests={"cpu": EXTRA_BIG_CPU, "memory": f"{EXTRA_BIG_MEMORY}Gi"},
                            limits={"cpu": EXTRA_BIG_CPU, "memory": f"{EXTRA_BIG_MEMORY}Gi"})
                    )
                ],
            )
        ),
    }
    
    ...
    
    second_level_model = task(
        task_id=f"2lvl_{model_name.split('.')[-1]}_{combiner_suffix}",
        executor_config=extra_big_executor_config
    )(fit_predict_second_level_model)(
        artifacts=artifacts,
        model_name=f"{model_name}_{combiner_suffix}",
        k=k,
        train_path=combined_train_path,
        first_level_predicts_path=combined_predicts_path,
        second_model_type=SECOND_LEVELS_MODELS_PARAMS[model_name]["second_model_type"],
        second_model_params=SECOND_LEVELS_MODELS_PARAMS[model_name]["second_model_params"],
        second_model_config_path=SECOND_LEVELS_MODELS_CONFIGS.get(model_name, None),
        cpu=EXTRA_BIG_CPU,
        memory=EXTRA_BIG_MEMORY
    )
    ```
    
    Here **extra_big_executor_config** is a patch being applied to [the default pod template](#pod-template), 
    **fit_predict_second_level_model** is a regular python function implemnting business logic, 
    **second_level_model** is an instance of Airflow Task used to build an Airflow Workflow.

3.  To send wheel and jar files to the cluster, use the command:
   ```shell
      ./experiments/bin/replayctl upd-wheels-and-jars
   ```

    This command will copy your wheel and jar files to the server so they can be used later in the `SparkSubmitOperator`.

### <a name="experiments"></a>How to run experiments with Airflow.

1.  Run the command: 
    ```shell
      ./experiments/bin/replayctl airflow-port-forward
    ```
    
    It should create a local proxy that connects to web UI of airflow.
   

2.  Go to a browser and open url http://localhost:8080 to access Airflow WebUI.


3.  Build and push an image for airflow executor (the image used by spawned pods on Kubernetes) with the command:
    ```shell
      ./experiments/bin/replayctl build-airflow-image
      ./experiments/bin/replayctl push-airflow-image
    ```


4. Sync dags files and their dependencies used by airflow itself:
    ```shell
      ./experiments/bin/replayctl sync-dags
    ```


5.  Go to webUI and start your dag. 
    See tutorials about how to do that here: 
    https://airflow.apache.org/docs/apache-airflow/stable/tutorial/fundamentals.html


### <a name="backfilling"></a>Workflow structure update and backfilling
TODO: backfilling
```shell
    airflow dags backfill -v --start-date=2022-12-07 -t second_level_lama_single_lgb -i 2stage_ml1m_alswrap
```

### <a name="external-services"></a>External and supplementary services to support workload executing

1.  [YARN cluster](http://node2.bdcl:32524/) is a cluster manager that executes Spark Applications in a distributed mode.
    
    This web ui may also be alternatively accessed through *kubectl* by the [link](http://localhost:8088/):  

    ```shell
      kubectl -n spark-lama-exps port-forward svc/yarn-resourcemanager 8088:8088
    ```
   
    **Important Note:** the port in the link may be a subject to change, 
    so the alternative way may be sometimes more preferable.


2.  [Spark History Server](http://node2.bdcl:31504/) is a web service that provides acces to Spark Web UI 
    of executing Spark applications and, what is even more important, to already finished or failed applications 
    thus providing capabilities to investigate and debug crashes and performance issues.

    This web ui may also be alternatively accessed through *kubectl* by the [link](http://localhost:18080/):  

    ```shell
      kubectl -n spark-lama-exps port-forward svc/yarn-spark-history-server 18080:18080
    ```
   
    **Important Note:** the port in the link may be a subject to change, 
    so the alternative way may be sometimes more preferable.


3.  [HDFS](http://node21.bdcl:9870/). An instance of HDFS being used in experiments along side nfs based on ESS.

    This web ui may also be alternatively accessed through *kubectl* by the [link](http://localhost:9870/):  

    ```shell
      kubectl -n hdfs-3 port-forward svc/hdfs-namenode 9870:9870
    ```
   
    **Important Note:** the port in the link may be a subject to change, 
    so the alternative way may be sometimes more preferable.

3.  [MLFlow](http://node2.bdcl:8811/) is a web service for reporting metrics and parameters of 
    running and finished experiments. Thus also providing capabilities for investigating and debugging. 


4.  [MLFlow-2](http://node2.bdcl:8822/) The second completely independent instance of MLFlow service. 
    One may freely use any of the instances, but one better prefer to use the first one 
    to simplify search for results on later stages.
    
5.  The project directory on ESS is located on **/mnt/ess_storage/DN_1/storage/SLAMA/kaggle_used_cars_dataset**. 
