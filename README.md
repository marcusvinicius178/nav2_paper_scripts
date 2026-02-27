# NAV2_Paper_Scripts

Scripts e pipeline para avaliação de trajetórias do Nav2 (ROS 2 Jazzy) com rosbags em MCAP, incluindo:

- Métricas primárias baseadas no planner do Nav2 (preferência: `/plan`, `/plan_smoothed`, `/transformed_global_plan`)
- Diagnóstico visual por run
- Métricas secundárias de “human-likeness” usando GPS bruto (`/gps/fix`) comparando simulação vs Ground Truth humano, com origem comum e correção automática de variante (ex.: `flip_y`)
- Execução batch para processar combinações completas (cenário, planner, controller, velocidade)

## Estrutura esperada

A raiz é o diretório:

`/home/marcus/NAV2_Paper_Scripts`

Estrutura recomendada:

```
NAV2_Paper_Scripts/
  RUNS/
    Scenario_1_Reta/
      NAVFN_MPPI_Vel_20/
        Navfn_mppi_1/
        ...
        Navfn_mppi_10/
        Output_charts_diagnostic/
        Output_metrics_assessment/
      NAVFN_RPP_Vel_20/
      SMAC_MPPI_Vel_25/
      ...
    Scenario_2_Reta_Arco_20/
    Scenario_3_Reta_Arco_40/
  gt_from_bag_outputs_aligned/
    gt_mean_references/
      scenario1_vel20_GTmean_rigidAligned_from_bag.csv
      scenario1_vel25_GTmean_rigidAligned_from_bag.csv
      scenario2_vel20_GTmean_rigidAligned_from_bag.csv
      ...
    per_bag/
      1-Scenario_reta_arco_Ground_truth/.../gps_fix_latlon.csv
      2-Scenario_reta_arco_20_Ground_truth/.../gps_fix_latlon.csv
      3-Scenario_reta_arco_40_Ground_truth/.../gps_fix_latlon.csv
  debug_raw_gps_waypoint_alignment.py
  eval_nav2_one_combination.py
  plot_nav2_run_diagnostics.py
  run_assessment_batch.py
  scenario1_waypoints_debug.yaml
  scenario2_waypoints_debug.yaml
  scenario3_waypoints_debug.yaml
```

Observação: as pastas `Output_charts_diagnostic/` e `Output_metrics_assessment/` são criadas automaticamente pelos scripts quando você não passa `--out-dir`.

## Requisitos

- Ubuntu com ROS 2 Jazzy instalado
- Python 3.12 (Jazzy) e dependências python básicas (numpy, matplotlib)
- `rosbag2_py` acessível via ambiente do ROS 2

Sempre rode:

```bash
source /opt/ros/jazzy/setup.bash
```

Opcional: criar um venv só para libs não-ROS, mas não é obrigatório.

## Scripts principais

### 1) Diagnóstico visual e figure “paper” (human GT GPS vs sim GPS)

Script:
- `debug_raw_gps_waypoint_alignment.py`

Ele faz:
- projeta GPS bruto (`/gps/fix`) para XY usando origem comum
- aplica correções simples (ex.: `flip_y`) e start-alignment
- gera figuras do paper com:
  - GT mean em destaque
  - envelope das runs em cinza
  - run representativa em destaque
- pode salvar figuras por run com `--save-per-run-plots`

#### Exemplo (Scenario 1, NavFn+MPPI, Vel 20)

```bash
cd /home/marcus/NAV2_Paper_Scripts
source /opt/ros/jazzy/setup.bash

python3 debug_raw_gps_waypoint_alignment.py \
  --sim-parent-dir /home/marcus/NAV2_Paper_Scripts/RUNS/Scenario_1_Reta/NAVFN_MPPI_Vel_20 \
  --sim-glob 'Navfn_mppi_*' \
  --gt-csv-glob '/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/per_bag/1-Scenario_reta_arco_Ground_truth/*vel_20*/gps_fix_latlon.csv' \
  --waypoint-file /home/marcus/NAV2_Paper_Scripts/scenario1_waypoints_debug.yaml \
  --save-per-run-plots
```

Outputs (por padrão):
- `.../NAVFN_MPPI_Vel_20/Output_charts_diagnostic/`

---

### 2) Métricas primárias e human-likeness (tabelas)

Script:
- `eval_nav2_one_combination.py`

Ele gera:
- `per_run_metrics.csv`
- `aggregated_metrics.csv`
- `primary_table_row.txt`
- `secondary_table_row.txt`
- `humanlikeness_table_row.txt`

Métricas primárias
- referência: caminho do planner (preferência: `/plan`, `/plan_smoothed`, `/transformed_global_plan`)
- sucesso geométrico: `success_geo = 1` se `PR >= 0.90`
- run válida para métricas contínuas: `valid_for_tracking = 1` se `PR >= 0.70`
- métricas contínuas calculadas só até o instante útil (primeiro PR>=0.90 ou PR máximo)

Human-likeness
- calculado só para runs `valid_for_tracking == 1`
- compara execução da simulação (GPS bruto) vs GT humano (GPS bruto)
- usa origem comum do waypoint-file
- aplica melhor variante (ex.: `flip_y`) e start-alignment

#### Exemplo (Scenario 1, NavFn+MPPI, Vel 20)

```bash
cd /home/marcus/NAV2_Paper_Scripts
source /opt/ros/jazzy/setup.bash

python3 eval_nav2_one_combination.py \
  --runs-dir /home/marcus/NAV2_Paper_Scripts/RUNS/Scenario_1_Reta/NAVFN_MPPI_Vel_20 \
  --scenario-id 1 \
  --speed-kmh 20 \
  --planner-id NavFn \
  --controller-id MPPI \
  --gt-csv /home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/gt_mean_references/scenario1_vel20_GTmean_rigidAligned_from_bag.csv \
  --gt-gps-csv-glob '/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/per_bag/1-Scenario_reta_arco_Ground_truth/*vel_20*/gps_fix_latlon.csv' \
  --waypoint-file /home/marcus/NAV2_Paper_Scripts/scenario1_waypoints_debug.yaml
```

Outputs (por padrão):
- `.../NAVFN_MPPI_Vel_20/Output_metrics_assessment/`

Observação: `--gt-csv` é aceito por compatibilidade, mas pode ser ignorado para métricas dependendo da versão do script.

---

### 3) Diagnóstico visual por run (planner vs execução em frame local)

Script:
- `plot_nav2_run_diagnostics.py`

Exemplo (uma run específica):

```bash
cd /home/marcus/NAV2_Paper_Scripts
source /opt/ros/jazzy/setup.bash

python3 plot_nav2_run_diagnostics.py \
  --bag-dir /home/marcus/NAV2_Paper_Scripts/RUNS/Scenario_1_Reta/NAVFN_MPPI_Vel_20/Navfn_mppi_1 \
  --gt-csv /home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/gt_mean_references/scenario1_vel20_GTmean_rigidAligned_from_bag.csv \
  --out-dir /home/marcus/NAV2_Paper_Scripts/RUNS/Scenario_1_Reta/NAVFN_MPPI_Vel_20/Navfn_mppi_1/diagnostics
```

---

## Execução batch (rodar charts + métricas de uma vez)

Script:
- `run_assessment_batch.py`

Ele percorre todos os diretórios de combinações dentro de um cenário e chama:
1. `debug_raw_gps_waypoint_alignment.py`
2. `eval_nav2_one_combination.py`

### Exemplo: rodar tudo do Scenario 1

```bash
cd /home/marcus/NAV2_Paper_Scripts
source /opt/ros/jazzy/setup.bash

python3 run_assessment_batch.py \
  --runs-root /home/marcus/NAV2_Paper_Scripts/RUNS \
  --scenario Scenario_1_Reta:1:/home/marcus/NAV2_Paper_Scripts/scenario1_waypoints_debug.yaml:'/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/per_bag/1-Scenario_reta_arco_Ground_truth/*vel_*/gps_fix_latlon.csv'
```

### Rodar só um subconjunto de combinações

Ex.: só `NAVFN_MPPI_Vel_*`

```bash
python3 run_assessment_batch.py \
  --runs-root /home/marcus/NAV2_Paper_Scripts/RUNS \
  --scenario Scenario_1_Reta:1:/home/marcus/NAV2_Paper_Scripts/scenario1_waypoints_debug.yaml:'/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/per_bag/1-Scenario_reta_arco_Ground_truth/*vel_*/gps_fix_latlon.csv' \
  --only-glob 'NAVFN_MPPI_Vel_*'
```

### Dry-run (só imprime os comandos)

```bash
python3 run_assessment_batch.py \
  --runs-root /home/marcus/NAV2_Paper_Scripts/RUNS \
  --scenario Scenario_1_Reta:1:/home/marcus/NAV2_Paper_Scripts/scenario1_waypoints_debug.yaml:'/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/per_bag/1-Scenario_reta_arco_Ground_truth/*vel_*/gps_fix_latlon.csv' \
  --dry-run
```

### Continuar mesmo se alguma combinação falhar

```bash
python3 run_assessment_batch.py \
  --runs-root /home/marcus/NAV2_Paper_Scripts/RUNS \
  --scenario Scenario_1_Reta:1:/home/marcus/NAV2_Paper_Scripts/scenario1_waypoints_debug.yaml:'/home/marcus/NAV2_Paper_Scripts/gt_from_bag_outputs_aligned/per_bag/1-Scenario_reta_arco_Ground_truth/*vel_*/gps_fix_latlon.csv' \
  --continue-on-error
```

---

## Subir para o GitHub (passo a passo)

### 1) Inicializar repositório (se ainda não for repo)

```bash
cd /home/marcus/NAV2_Paper_Scripts
git init
```

### 2) Criar um `.gitignore` (recomendado)

Crie um arquivo:

`/home/marcus/NAV2_Paper_Scripts/.gitignore`

Sugestão:

```
__pycache__/
*.pyc
*.png
*.csv
*.txt
*.mcap
*.zstd

RUNS/
outputs_*/
debug_*/
Output_charts_diagnostic/
Output_metrics_assessment/
*/Output_charts_diagnostic/
*/Output_metrics_assessment/

nav2_paper_scripts_sem_RUNS_waypoints.tar.gz
```

### 3) Adicionar arquivos e commitar

```bash
git add README.md .gitignore *.py scenario*.yaml gt_from_bag_outputs_aligned/README_summary.txt
git commit -m "Initial release: Nav2 trajectory evaluation scripts"
```

Atenção: inclua somente arquivos que você quer publicar.

### 4) Criar o repositório no GitHub

No GitHub:
- New repository
- Escolha nome, ex.: `nav2_paper_scripts`
- Não precisa inicializar com README (já temos local)

### 5) Vincular remote e fazer push

Substitua `<URL_DO_REPO>` pelo URL do seu repositório (HTTPS ou SSH):

```bash
git branch -M main
git remote add origin <URL_DO_REPO>
git push -u origin main
```

Se usar HTTPS e pedir token: use um GitHub Personal Access Token.
Se usar SSH: configure suas chaves SSH.

---

## Checklist rápido antes de publicar

- Remover qualquer dado sensível (paths pessoais, nomes, etc. se necessário)
- Garantir que `RUNS/` e rosbags não entram no GitHub
- Conferir se waypoint YAML e glob de GT não vazam dados indevidos (normalmente ok)
- Rodar um exemplo do README para validar
