## HOW TO LAUNCH WEB.APP
## TO DOS
- Run a optimized process through optuna
- full descriptive analytics,Build another page statsitic : distrib pf hapiness : are zyou happy? over the last month 3.
    - What is the activiy that is done the most, and what is the impact on the overall target? If you do something a lot but it does not make you happy what is the purpose?
    - how to categorise it 1 i good, 0 is neutral, -1 is bad.
- Additional features :  
    - interaction sociale (friends)
    - temps pour soi
    - lecture 
    - faire une bonne
    - altercation (dispuste/extermal factor)
- For deeper analysis:
    - make output between 0 1 2 recalculate a an average socre and scale it between 0 and 1. (divide by max-min)
- Make a more complex model with Optuna and hyperparameter tuning when you have more data.
- Make appear the decision Tree as an exemple. (explain which algorithm is behind)
- Deploy app online https://towardsdatascience.com/3-easy-ways-to-deploy-your-streamlit-web-app-online-7c88bb1024b1


HOW TO HACK YOUR WELLNESS POWERED BY A.I
==========

Environment setup
-----------------

#. Clone the repository::

    git clone ...

#. Create a virtual environment::

    python3 -m venv venv
    source venv/bin/activate

#. Install the package in editable mode::

    pip install -e .

#. Install dev dependencies::

    pip install -r requirements.txt
   
#. Go in the folder and run the app::

    streamlit run main_page.py