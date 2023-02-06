###Table of content
1. [General info](#general-info)

2. [Important files](#important-files)

3. [Short list of important supplementary commands](#sup-commands)

4. [How to deploy DAGs to an Airflow instance](#deploy-dags)

5. [How to run experiments with Airflow](#experiments)

6. [Workflow structure update and backfilling](#backfilling)

7. [External and supplementary services to support workload executing](#external-services)

### <a name="general-info"></a>General information.
TODO: the big picture of how it works

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
   

2.  To execute a task Airflow is configured to use Kubernetes executor that runs tasks as separate pods 
    inside Airflow namespace on Kubernetes. 
    This is also true for tasks that submit Spark jobs to a standalone YARN cluster. In such a case, 
    the pod will execute spark-submit script that starts a job on YARN and 
    is constantly reporting status of the job until it is finished or failed.
    
    Each pod is created from the following template which is a content of pod_template_file.yaml 
    as it was mentioned earlier. 
    The current version of the pod template 
    (be aware that the actual pod template may differ one better check it for himself using the command above):
    
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
    
    To alter the pod's configuration one may apply ad-hoc patch in the dag file <TODO: dag file> the following way: 
    <TODO: patch>

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
```shell
    airflow dags backfill -v --start-date=2022-12-07 -t second_level_lama_single_lgb -i 2stage_ml1m_alswrap
```

### <a name="external-services"></a>External and supplementary services to support workload executing

1. TODO: YARN

2. TODO: spark-history-server
