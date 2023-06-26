## K8S configuration files

### Redis
`redis-kube-pv-pvc.yml` - creates PV and PVC to persist data from redis:
```bash
kubectl -n <namespace> apply -f ./redis-kube-pv-pvc.yml
```

`redis-kube.yml` - creates deployment and service:
```bash
kubectl -n <namespace> apply -f ./redis-kube.yml
```


Access to redis
```python
import redis

redis_db = redis.Redis(host='redis-slama.airflow')
```