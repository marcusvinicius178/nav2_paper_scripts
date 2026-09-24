# Navigation2 Benchmark Analysis Scripts

Analysis and reproducibility code associated with the manuscript:

**“Heavy-Duty Agricultural Truck Navigation: Geometry-Dependent Planner–Controller Trade-offs in a Human-Referenced ROS 2 Navigation2 Benchmark.”**

This repository contains the processing pipelines used to analyze ROS 2 Navigation2 experiments performed with a heavy-duty Ackermann-steered agricultural truck.

The code covers three complementary parts of the study:

- simulation benchmark analysis;
- professional human-driver reference analysis;
- field trajectory and obstacle-avoidance analysis.

The evaluated navigation configurations include NavFn and SMAC global planners combined with MPPI and Regulated Pure Pursuit (RPP) controllers.

## Related resources

| Resource | Purpose |
| --- | --- |
| [gps_truck_nav](https://github.com/marcusvinicius178/gps_truck_nav) | Vehicle integration, ROS 2 Navigation2 configuration, Gazebo models and simulation support |
| This repository | Analysis, metrics, tables, figures and reproducibility scripts |
| [Zenodo dataset – 10.5281/zenodo.22864167](https://doi.org/10.5281/zenodo.22864167) | Raw ROS 2 recordings used by the study |

Raw rosbags are intentionally not stored in GitHub.

## Repository structure

```text
nav2_paper_scripts/
├── configs/
│   └── field_audit_v3.yaml
├── scripts/
│   ├── simulation/
│   │   ├── eval_nav2_one_combination.py
│   │   ├── run_supplementary_tables_pipeline.py
│   │   ├── build_supplementary_tables.py
│   │   ├── plot_corrected_figures_5_7.py
│   │   ├── debug_raw_gps_waypoint_alignment.py
│   │   └── test_plan_cross_track_v2.py
│   ├── human_reference/
│   │   ├── build_gt_final_table_v3.py
│   │   └── plot_human_reference_repeatability.py
│   └── field/
│       ├── recompute_field_metrics_from_bags_v3.py
│       ├── plot_field_obstacle_and_trajectory_from_bags.py
│       ├── run_field_audit_pipeline_v3.py
│       ├── audit_field_rosbags_v3.py
│       └── field_audit_common_v3.py
├── tests/
├── requirements_v3.txt
└── README.md
```

Additional earlier scripts are retained where useful for comparison and reproduction of intermediate processing steps.

## Environment

The analysis code uses Python 3 and, for scripts that read ROS 2 bags directly, the ROS 2 Python libraries and `rosbag2_py`.

The development environment used ROS 2 Jazzy for bag-processing utilities.

Source ROS before running scripts that access MCAP/rosbag2 data:

```bash
source /opt/ros/jazzy/setup.bash
```

Install the additional Python dependencies with:

```bash
python3 -m pip install -r requirements_v3.txt
```

## Data

The repository does not assume a fixed filesystem location for the experimental data.

Download or mount the ROS 2 recordings and provide the corresponding paths to the analysis scripts through their command-line arguments.

For the exact options supported by each pipeline:

```bash
python3 scripts/simulation/run_supplementary_tables_pipeline.py --help
python3 scripts/simulation/eval_nav2_one_combination.py --help
python3 scripts/human_reference/build_gt_final_table_v3.py --help
python3 scripts/field/run_field_audit_pipeline_v3.py --help
```

## Simulation analysis

The principal simulation-processing code is under:

```text
scripts/simulation/
```

The workflow includes extraction of trajectory information from ROS 2 recordings, calculation of benchmark metrics, aggregation across repeated runs, generation of supplementary tables, and generation of figures used in the manuscript analysis.

The main entry points are:

```text
run_supplementary_tables_pipeline.py
eval_nav2_one_combination.py
build_supplementary_tables.py
plot_corrected_figures_5_7.py
```

The simulation benchmark contains the planner/controller combinations evaluated for scenarios S1, S2 and S3 and their corresponding speed conditions.

## Human-reference analysis

Human-driven reference trajectories are processed under:

```text
scripts/human_reference/
```

These scripts support trajectory repeatability analysis and construction of the human-reference quantities used for comparison with the autonomous navigation runs.

The current final-table and visualization tools include:

```text
build_gt_final_table_v3.py
plot_human_reference_repeatability.py
```

## Field analysis

Field-processing scripts are under:

```text
scripts/field/
```

They support extraction and recomputation of trajectory, heading, clearance, timing and related quantities from the field ROS 2 recordings.

The field dataset contains the human reference and the autonomous planner/controller demonstrations described in the manuscript.

## Testing

Basic regression tests are provided under `tests/`.

Run them from the repository root with:

```bash
python3 -m unittest discover -s tests -v
```

Python syntax can also be checked with:

```bash
find scripts tests -type f -name '*.py' -print0 \
  | xargs -0 -n1 python3 -m py_compile
```

## Reproducibility organization

The research artifacts are separated intentionally:

```text
gps_truck_nav
    vehicle integration and simulation configuration

nav2_paper_scripts
    analysis, metrics, figures and tables

Zenodo dataset
    raw ROS 2 recordings
```

This keeps large experimental recordings outside Git while keeping the processing code openly inspectable and reusable.

## Citation

```bibtex
@unpublished{carvalho2026trucknavigation,
  title = {Heavy-Duty Agricultural Truck Navigation:
           Geometry-Dependent Planner--Controller Trade-offs
           in a Human-Referenced ROS 2 Navigation2 Benchmark},
  author = {Carvalho, Marcus Vin{\'i}cius Leal de
            and Yoshioka, Leopoldo Rideki
            and Justo, Jo{\~a}o Francisco
            and Silva, Antonio Marcos da},
  year = {2026},
  note = {Manuscript submitted to Computers and Electronics in Agriculture}
}
```

### Authors

- Marcus Vinícius Leal de Carvalho
- Leopoldo Rideki Yoshioka
- João Francisco Justo
- Antonio Marcos da Silva

Corresponding author: **Marcus Vinícius Leal de Carvalho**

Email: **marcusvini178@usp.br**
