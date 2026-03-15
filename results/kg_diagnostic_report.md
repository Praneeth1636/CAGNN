# Knowledge Graph Diagnostic Report

## Executive Summary
- **Health score:** 16.3/100
- **Bottleneck edges:** 34
- **Multi-hop vulnerability:** 100.0%

## Top Bottlenecks
| # | Source | Relation | Target | Curvature | Impact |
|---|--------|----------|--------|------------|--------|
| 1 | Project Beta | depends_on | Project Delta | -0.167 | 0.2081 |
| 2 | ML Team | depends_on | Project Beta | -0.524 | 0.2003 |
| 3 | Frontend Team | depends_on | Project Delta | -0.417 | 0.1939 |
| 4 | Security Team | depends_on | Project Alpha | -0.596 | 0.1917 |
| 5 | ML Team | depends_on | Project Alpha | -0.500 | 0.1650 |
| 6 | Frank | leads | Backend Team | -0.425 | 0.1108 |
| 7 | Data Team | depends_on | Project Beta | -0.524 | 0.1079 |
| 8 | DevOps Team | depends_on | Project Alpha | -0.417 | 0.0947 |
| 9 | Project Alpha | depends_on | Project Gamma | -0.292 | 0.0879 |
| 10 | ML Team | works_in | Bridge Person | -0.393 | 0.0853 |

## Bridge Entities
- **ML Team** (Team): curvature=-0.184
- **Project Alpha** (Project): curvature=-0.451
- **Data Team** (Team): curvature=-0.108
- **Project Beta** (Project): curvature=-0.405
- **Backend Team** (Team): curvature=-0.166
- **DevOps Team** (Team): curvature=-0.112
- **Project Delta** (Project): curvature=-0.292
- **HQ** (Location): curvature=-0.162
- **Frank** (Person): curvature=-0.106
- **Python** (Technology): curvature=-0.383

## Knowledge Islands
- Cluster 0: 22 nodes
- Cluster 1: 8 nodes
- Cluster 2: 13 nodes
- Cluster 3: 11 nodes
- Cluster 4: 2 nodes
- Cluster 5: 11 nodes
- Cluster 6: 1 nodes
- Cluster 7: 2 nodes
- Cluster 8: 2 nodes
- Cluster 9: 1 nodes
- Cluster 10: 1 nodes
- Cluster 11: 1 nodes
- Cluster 12: 4 nodes
- Cluster 13: 1 nodes

## Recommended Fixes
| Source | Target | Relation | Priority |
|--------|--------|----------|----------|
| Alice | DevOps Team | leads | low |
| Alice | Project Alpha | leads | low |
| Alice | Project Beta | leads | low |
| Alice | Python | leads | low |
| Alice | HQ | leads | high |
| Bob | Dave | works_in | low |
| Bob | Project Alpha | works_in | low |
| Bob | Project Beta | works_in | low |
| Bob | Python | works_in | low |
| Bob | HQ | works_in | high |
| Bob | Bridge Person | works_in | low |
| Dave | ML Team | leads | low |
| Dave | Project Alpha | leads | low |
| Dave | AWS | leads | low |
| Dave | Cloud DC1 | leads | low |

## Figures
- Overview: `figures/kg_overview.png`
- Health dashboard: `figures/kg_health_dashboard.png`