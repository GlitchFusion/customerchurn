churn_prediction/
│
├── .gitignore
├── README.md                           
├── requirements.txt                     
├── main.py                             
│
├── config/                              
│   └── configs.py                       
│
├── src/                                 
│   ├── data/
│   │   ├── loader.py                    
│   │   └── preprocessor.py              
│   │
│   ├── models/
│   │   └── custom_logistic.py           
│   │
│   ├── training/
│   │   └── trainer.py                   
│   │
│   ├── evaluation/
│   │   ├── metrics.py                   
│   │   ├── visualizer.py                
│   │   └── benchmark.py                 
│   │
│   └── utils/
│       ├── logger.py                    
│       └── io_helpers.py                
│
├── tests/                               
│
├── deployment/                          
│   ├── app.py                           
│   ├── static/                          
│   └── templates/                       
│
├── data/                                
│   ├── raw/
│   │   └── Telco-Customer-Churn-Data.csv.csv              
│
├── models/                              
│
├── reports/                             
│   ├── figures/
│
└── logs/                               
    └── training.log