# MVP — Alocação eficiente de autocarros e metrobus

Este projeto cria um MVP para simular e otimizar alocação de frota por linha e por hora.

## to do :
- dados: corrigir e limprar dados metrobus: criar shapes.yxy, corrigir agency, feed_info, etc

## O que o MVP faz

- Lê GTFS (`routes.txt`, `trips.txt`, `stop_times.txt`, `stops.txt`)
- Usa procura real por hora (`demand_csv`) **ou** gera procura sintética
- Otimiza número de veículos por linha/hora com `OR-Tools`
- Gera saídas comparáveis para decisão operacional
- Renderiza mapa HTML com linhas e paragens

## Estrutura

- `run_mvp.py`: ponto de entrada
- `src/smtuc_mvp/gtfs.py`: ingestão GTFS + dados sintéticos
- `src/smtuc_mvp/optimizer.py`: modelo de otimização e KPIs
- `src/smtuc_mvp/map_viz.py`: visualização de rede
- `src/smtuc_mvp/pipeline.py`: orquestração completa

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Execução rápida (sem GTFS, dados sintéticos)

```bash
python run_mvp.py
```

## Execução com GTFS real

```bash
python run_mvp.py --gtfs-dir "C:/dados/gtfs_smtuc" --fleet 85 --bus-capacity 72 --metrobus-capacity 120
```

## Execução com procura observada

Cria um CSV com colunas:

- `route_id`
- `hour` (0-23)
- `demand_pax`

Exemplo:

```csv
route_id,hour,demand_pax
R1,8,520
R1,9,430
MB,8,980
```

Depois roda:

```bash
python run_mvp.py --gtfs-dir "C:/dados/gtfs_smtuc" --demand-csv "C:/dados/demand.csv"
```

## Outputs

No diretório `outputs/`:

- `demand_route_hour.csv`
- `allocation_route_hour.csv`
- `kpis.json`
- `network_map.html`

## Próximos upgrades recomendados

- Inserir tempos reais GPS por período
- Calibrar procura por OD e não apenas por linha/hora
- Adicionar restrições de motorista/turno/depot
- Rodar cenários com Monte Carlo para robustez
