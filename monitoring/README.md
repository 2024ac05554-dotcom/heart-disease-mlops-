\# Monitoring Setup



Run alongside the API (`uvicorn app.main:app --port 8000`):



\## 1. Create a shared Docker network

docker network create monitoring-net



\## 2. Run Prometheus (scrapes the API's /metrics endpoint)

docker run -d --name prometheus --network monitoring-net -p 9090:9090 -v "${PWD}/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus



\## 3. Run Grafana

docker run -d --name grafana --network monitoring-net -p 3000:3000 grafana/grafana



\## 4. Access

\- Prometheus: http://localhost:9090 (Status > Targets to confirm scraping)

\- Grafana: http://localhost:3000 (admin/admin), add Prometheus data source at http://prometheus:9090

\- Dashboard: "Heart Disease API Monitoring" — panel showing predict\_requests\_total over time



