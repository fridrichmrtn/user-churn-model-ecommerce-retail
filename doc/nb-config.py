# jupyter notebook to markdown config
# usage> jupyter nbconvert --execute --config nb-config.py
c = get_config()
c.NbConvertApp.notebooks = ["user-churn-benchmark.ipynb"]
#c.NbConvertApp.notebooks = ["user-churn-model.ipynb", "user-churn-exploration.ipynb", "user-churn-benchmark.ipynb", "user-churn-features.ipynb"]

#c.NbConvertApp.notebooks = ["user-churn-features.ipynb"]
c.NbConvertApp.export_format = "markdown"
c.NbConvertApp.output_files_dir = "img/{notebook_name}/"
c.ExecutePreprocessor.timeout = 100000