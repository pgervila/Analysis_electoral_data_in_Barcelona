import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# LOAD BARRIS DATA ( POPULATION + ELECTORAL MERGED DATA )

# TODO : index_col specification wrong ??
df_el_CAT_bcn_2015_MERGED = pd.read_csv("saved_data/el_CAT_barris_bcn_2015_MERGED.csv", index_col=0)

df_el_gen_bcn_2015_MERGED = pd.read_csv("saved_data/el_gen_barris_bcn_2015_MERGED.csv", index_col=0)

df_el_gen_bcn_2016_MERGED = pd.read_csv("saved_data/el_gen_barris_bcn_2016_MERGED.csv", index_col=0)

df_el_CAT_bcn_2017_MERGED = pd.read_csv("saved_data/el_CAT_barris_bcn_2017_MERGED.csv", index_col=0)


# DEFINE PLOTTING FUNCTIONS

# TODO : make function more general to plot Union, DretD, etc ...
def plot_indy_vs_var(data, label_x, size_dots=25):
    """
        Function to visualize independence support as a function
        of a population or electoral variable via scatter plot
        Args:
            * data: pandas dataframe. It must contain  merged pop + electoral data
            * label_x : string. Column name to identify independent variable

    """
    data = data.drop(data[data['Indep_pct'] == 0].index)
    fig, ax = plt.subplots()
    ax.scatter(x=data[label_x], y=data['Indep_pct'],
               c=np.where(data['Indep_pct'] >= 50, 'blue', 'red'),
               alpha=0.7, s=size_dots)
    import matplotlib.ticker as ticker
    ax.set_xlabel(label_x, fontsize=14)
    ax.set_ylabel('Indep support %', fontsize=14)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(linestyle='dotted', linewidth=2)
    ax.set_title('Independence support vs ' + label_x + ' in Barcelona',
                 family='serif', fontsize=18)
    plt.savefig('Indy_support_vs_' + label_x + '_in_Barcelona.png')


# LAYOUT for REGRESISON ANALYSIS
import statsmodels.api as sm
import statsmodels.formula.api as smf
result_ols = smf.ols('Indep_pct ~ NascutsRestaEstat_Votants_ratio + np.log(Effective_Educ_univ_pct) + Tot_Atur_pct ',
                     data = df_el_gen_bcn_2015_MERGED).fit()
print(result_ols.summary())



# CATALAN ELECTIONS 21-D 2017 ( BARRIS )


# CATALAN ELECTIONS 21-D 2017 ( SECCIO CENSAL )

df_el_CAT_bcn_SC_2017_MERGED = pd.read_csv("saved_data/el_CAT_bcn_SC_2017_MERGED.csv")
# data stats
df_el_CAT_bcn_SC_2017_MERGED.filter(regex='pct').describe()