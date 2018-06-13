import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['figure.figsize'] = (15,10)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
from adjustText import adjust_text



# URLs FOR ELECTORAL DATA

# BARRIS
url_el_cat_bcn_2015_barris = 'http://www.bcn.cat/estadistica/catala/dades/barris/telec/aut/a2015.htm'
url_el_cat_bcn_2017_barris = "http://www.bcn.cat/estadistica/catala/dades/inf/ele/ele42/A18.htm"
url_el_gen_bcn_2015_barris_bcn = "http://www.bcn.cat/estadistica/catala/dades//inf/ele/ele38/A18.htm"
url_el_gen_bcn_2016_barris_bcn = "http://www.bcn.cat/estadistica/catala/dades/telec/gen/gen16/A18.htm"

# URLs for data on 'SECCIONS CENSALS' ( smallest census districts)
url_el_cat_bcn_sc_2015 = "http://www.bcn.cat/estadistica/catala/dades/inf/ele/ele36/A110.htm"
url_el_gen_bcn_sc_2015 = "http://www.bcn.cat/estadistica/catala/dades//inf/ele/ele38/A110.htm"
url_el_gen_bcn_sc_2016 = "http://www.bcn.cat/estadistica/catala/dades/inf/ele/ele40/A110.htm"
url_el_cat_bcn_sc_2017 = "http://www.bcn.cat/estadistica/catala/dades/inf/ele/ele42/A110.htm"


# URLs for POPULATION DATA

# BARRIS
url_lloc_naix_barris_bcn = "http://www.bcn.cat/estadistica/catala/dades/tpob/pad/cens/a2011/llocna/lloc05.htm"
url_nacional_barris_bcn = "http://www.bcn.cat/estadistica/catala/dades/barris/tpob/pad/a2015/cp14.htm"
url_renda_fam_barris_bcn = "http://www.bcn.cat/estadistica/catala/dades/barris/economia/renda/rdfamiliar/a2014.htm"
url_invalid_barris_bcn = "http://www.bcn.cat/estadistica/catala/dades/barris/tvida/discapacitats/t0414.htm"
url_nivell_acad_barris_bcn = "http://www.bcn.cat/estadistica/catala/dades/barris/tpob/pad/padro/a2015/cp27.htm"
url_nivell_atur_barris_bcn = "http://www.bcn.cat/estadistica/catala/dades/barris/ttreball/atur/durada/durbargen.htm"
url_edat_barris_bcn = "http://www.bcn.cat/estadistica/catala/dades/barris/tpob/pad/padro/a2016/cp05.htm"

# SECCIONS CENSALS
url_lloc_naix_sc_bcn = "http://www.bcn.cat/estadistica/catala/dades/tpob/pad/padro/a2017/llocna/llocna11.htm"
url_nivell_acad_sc_bcn = "http://www.bcn.cat/estadistica/catala/dades/tpob/pad/padro/a2017/nivi/nivi11.htm"


# CORRESPONDENCE OF TERRITORIAL SUBDIVISIONS
# dataframe with correspondence info between all territorial subdivisions in Barcelona
df_divs = pd.read_csv('saved_data/bcn_divisions_corresp.csv')


# DEFINE FUNCTIONS TO SCRAPE DATA FROM URLs

def imp_data_barris_bcn(url, df_header=4, df_skiprows=[5, 6, 7, 8]):
    """
        Function to scrape HTML tables from Barcelona city hall web site for barris city jurisdiction
        by providing URL where the table can be visualized
        Header and skipped rows values might need to be tuned depending on table data
    """
    # scrape data
    df_barris = pd.read_html(url, header=df_header, skiprows=df_skiprows,
                             thousands='.', encoding='latin-1')[0]
    # keep only absolute vote
    df_barris.drop(df_barris.index[73:], inplace=True)
    #
    if "Dte. Barris" in df_barris.columns:
        df_barris[['DTE',
                   'BARRI',
                   'NOM_BARRI']] = df_barris["Dte. Barris"].str.split(pat=r"[.\s]", n=2, expand=True)
        df_barris.drop("Dte. Barris", axis = 1, inplace = True)
    elif "Dte." and "Barri" in df_barris.columns:
        df_barris[['BARRI','NOM_BARRI']] = df_barris["Barri"].str.split(pat=r"[.\s]", n=1, expand=True)
        df_barris.drop("Barri", axis = 1, inplace=True)
    else:
        df_barris[['DTE',
                   'BARRI',
                   'NOM_BARRI']] = df_barris["Dte."].str.split(pat=r"[.\s]", n=2, expand=True)
        df_barris.drop("Dte.", axis=1, inplace=True)
    df_barris = df_barris.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    return df_barris


def imp_data_sc_bcn(url, df_header=4, df_skiprows=[5, 6, 7, 8]):  # scrape data
    """ Function to scrape HTML table data from Barcelona Seccions Censal using url.
        Args:
            * url: string. URL for Barcelona city hall official web site
            * df_header: integer
            * df_skiprows: list of integers. Integers indicate rows that must be skipped when reading HTML table
        Output:
            * pandas dataframe with the requested electoral info
    """
    df_SC = pd.read_html(url,
                         header=df_header,
                         skiprows=df_skiprows,
                         thousands='.',
                         encoding='latin-1')[0]
    # drop unnecessary rows
    delta_rows = df_SC.shape[0] - 1068
    df_SC.drop(df_SC.index[-delta_rows:], inplace=True)
    df_SC[['DTE', 'SC']] = df_SC["Dte. SC"].str.split(pat=r"[.\s]",
                                                      n=2,
                                                      expand=True)
    df_SC.drop("Dte. SC", axis=1, inplace=True)
    df_SC = df_SC.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    return df_SC


def process_imp_data(data, rename_dict, party_names, groups_dict):
    """
        Function to add new columns to pandas dataframe scraped from URL
        Args:
            * data: pandas DataFrame.
            * rename_dict: dictionary. Used to rename specific columns. Keys are old names, values new ones
            * party_names: list of strings. Strings are the names of the political parties involved in electoral contest
            * groups_dict: dictionary. Keys are strings that define groups as defined with respect to independence
                ('Indep', 'DretD', 'Union'), values are lists of strings with the party names for each group
        Output:
            * Modified pandas DataFrame
    """

    data.rename(columns = rename_dict, inplace= True)
    # compute vote percentages
    for name in party_names:
            data[name + '_pct'] = (100 * data[name] /
                                   data['Votants']).round(decimals=2)
            data[name + '_pct_Electors'] = (100 * data[name] /
                                            data['Electors']).round(decimals=2)
    for key, parties in groups_dict.items():
        data[key] = data[parties].sum(axis=1)
        pct_list_keys = [x + "_pct" for x in parties]
        data[key + "_pct"] = data[pct_list_keys].sum(axis=1)
        pct_Electors_list_keys = [x + "_pct_Electors" for x in parties]
        data[key + "_pct_Electors"] = data[pct_Electors_list_keys].sum(axis=1)
    data['Participacio_pct'] = 100 * (data['Votants'] / data['Electors'])
    return data


# SCRAPE POPULATION DATA

# place of birth data ( BARRIS )

# scrape data
df_lloc_naix = imp_data_barris_bcn(url_lloc_naix_barris_bcn, df_header=4, df_skiprows=[5,6,7,8])
# process data
df_lloc_naix.rename(columns={"Catalunya": "CAT", "Resta de l'Estat": "Resta_Estat", "TOTAL":"TOTAL_BORN"},
                    inplace=True)
llocs = ['CAT','Resta_Estat','Estranger']
for lloc in llocs:
    df_lloc_naix[lloc + '_pct'] = (100 * df_lloc_naix[lloc] /
                                   df_lloc_naix['TOTAL_BORN']).round(decimals=2)
df_lloc_naix["NascutsRestaEstat_Votants_ratio"] = 100 * (df_lloc_naix["Resta_Estat"] /
                                                        (df_lloc_naix["Resta_Estat"] + df_lloc_naix["CAT"]))
# save data
df_lloc_naix.to_csv("data_barris_lloc_naix.csv")

# place of birth data ( Secció censal )

# scrape data
df_lloc_naix_SC = imp_data_sc_bcn(url_lloc_naix_sc_bcn)
# process data
df_lloc_naix_SC['CAT'] = df_lloc_naix_SC["Barcelona ciutat"] + df_lloc_naix_SC["Resta Catalunya"]
df_lloc_naix_SC.drop(['Barcelona ciutat', 'Resta Catalunya', "No consta"], axis=1, inplace=True)
df_lloc_naix_SC["Resta_Estat"] = df_lloc_naix_SC.drop(['TOTAL', 'CAT', 'Estranger', 'DTE', 'SC'], axis=1).sum(axis=1)
df_lloc_naix_SC = df_lloc_naix_SC[['TOTAL', 'CAT', "Resta_Estat", "Estranger", 'DTE', 'SC']]

df_lloc_naix_SC.rename(columns={"TOTAL":"TOTAL_BORN"}, inplace = True)
llocs = ['CAT', 'Resta_Estat', 'Estranger']
for lloc in llocs:
    df_lloc_naix_SC[lloc + '_pct'] = (100 * df_lloc_naix_SC[lloc] /
                                      df_lloc_naix_SC['TOTAL_BORN']).round(decimals=2)
df_lloc_naix_SC["NascutsRestaEstat_Votants_ratio"] = 100 * (df_lloc_naix_SC["Resta_Estat"] /
                                                            df_lloc_naix_SC[["Resta_Estat", "CAT"]].sum(axis=1))

# save data
df_lloc_naix_SC.to_csv("saved_data/data_lloc_naix_bcn_SC.csv", index=False)


# academic level data ( BARRIS )

# scrape data
df_acad_level = imp_data_barris_bcn(url_nivell_acad_barris_bcn)
# process data
df_acad_level.rename(columns={"Estudis primaris / certificat d'escolaritat / EGB": "Educ_primar",
                              "Batxillerat elemental / graduat escolar / ESO / FPI": "Educ_batx_elem",
                              "Batxillerat superior / BUP / COU / FPII / CFGM grau mitjà":"Educ_batx_sup",
                              "Estudis universitaris / CFGS grau superior":"Educ_univ",
                              "Sense estudis":"Educ_NO",
                              "TOTAL":"TOTAL_EDUC"},
                     inplace=True)
df_acad_level['Educ_univ_pct'] = 100*(df_acad_level['Educ_univ'] /
                                      df_acad_level['TOTAL_EDUC']).round(decimals=4)
df_acad_level['Educ_primar_pct'] = 100*(df_acad_level['Educ_primar'] /
                                        df_acad_level['TOTAL_EDUC']).round(decimals=4)
# save data
df_acad_level.to_csv("data_barris_nivell_academic.csv")

# Unemployment data ( BARRIS )

# scrape
df_unempl_level = imp_data_barris_bcn(url_nivell_atur_barris_bcn)
# process
df_unempl_level['Tot_Atur_pct'] = 100 * (df_unempl_level['Total Aturats'] /
                                         df_acad_level['TOTAL_EDUC']).round(decimals=4)
df_unempl_level['Tot_LT_Atur_pct'] = 100 * (df_unempl_level['Més de 12 mesos'] /
                                            df_acad_level['TOTAL_EDUC']).round(decimals=4)
df_unempl_level.to_csv("data_barris_atur.csv")

# age data ( BARRIS )

df_edats = imp_data_barris_bcn(url_edat_barris_bcn)
df_edats["Over_fifty_pct"] = 100 * df_edats.iloc[:, df_edats.columns.str.startswith(('50-','55-','60-','65-','70-','75-','80-',
                                                 '85-', '90-', '95'))].sum(axis=1)/df_edats['TOTAL']


# income level data ( BARRIS )
df_renda = imp_data_barris_bcn(url_renda_fam_barris_bcn)
df_renda.rename(columns={'Índex RFD Barcelona = 100': "Index_RFD"}, inplace=True)
df_renda.Index_RFD = df_renda.Index_RFD.str.replace(",", ".")
df_renda = df_renda.apply(lambda x: pd.to_numeric(x, errors='ignore'))
df_renda.set_value(10, "NOM_BARRI", ' el Poble Sec - AEI Parc Montjuïc')
df_renda.to_csv("data_barris_renda.csv")
a1 = np.where(df_renda['Index_RFD'] <= 75, 1, df_renda['Index_RFD'])
a2 = np.where((a1 > 75) & (a1 <= 100), 2, a1)
a3 = np.where((a2 > 100) & (a2 <= 150), 3, a2)
df_renda['Index_intervals'] = np.where((a3 > 150), 4, a3)


# MERGE ALL POPULATION DATA
df_popul_data_MERGED = pd.merge(df_lloc_naix, df_renda, on=['DTE','BARRI','NOM_BARRI'])

df_popul_data_MERGED = pd.merge(df_popul_data_MERGED, df_acad_level,
                                on=['DTE', 'BARRI', 'NOM_BARRI'])

df_popul_data_MERGED = pd.merge(df_popul_data_MERGED, df_unempl_level,
                                on=['DTE', 'BARRI', 'NOM_BARRI'])


# CATALAN ELECTIONS 27S 2015 ( BARRIS )

# scrape electoral data
url = 'http://www.bcn.cat/estadistica/catala/dades/barris/telec/aut/a2015.htm'
rename_dict = {'JxSí (1)': "JxSi", 'CatSíque- esPot (2)': "ECP", "C’s": "Cs"}
party_names = ['JxSi','Cs','PSC','CUP','PP','ECP']
groups_dict = {'Indep': ["JxSi", "CUP"], 'DretD': ["JxSi", "CUP", "ECP"], 'Union': ["Cs", "PP", "PSC"]}
data = imp_data_barris_bcn(url, df_header=6, df_skiprows=[7, 8, 9, 10])
# process data
df_el_CAT_bcn_2015 = process_imp_data(data, rename_dict, party_names, groups_dict)

# MERGE ELECTORAL DATA WITH POPULATION DATA

df_el_CAT_bcn_2015_MERGED = pd.merge(df_popul_data_MERGED, df_el_CAT_bcn_2015, on=['DTE','BARRI','NOM_BARRI'])


# Since academic data is not available by place of birth, we need to filter the bias from foreigners,
# that cannot vote and that do not hold degrees in most quarters.
# A quarter with a large proportion of uneducated foreigners would result into a low value
# of calculated degree density, whereas the ratio of degrees to voters might still be relatively high.
# We thus define the ratio of people with degrees to the people that are allowed to vote.
# This is not perfect, since in richer quarters most foreigners will probably have a degree,
# resulting into a higher than real relevant-for-voting degree density. However,
# in those quarters the percentage of foreigners is expected to be low or in any case
# much lower than in poorer quarters

df_el_CAT_bcn_2015_MERGED["Effective_Educ_univ_pct"] = (100 * df_el_CAT_bcn_2015_MERGED["Educ_univ"] /
                                                        df_el_CAT_bcn_2015["Electors"]).round(decimals=2)
# save data
df_el_CAT_bcn_2015_MERGED.to_csv("el_CAT_barris_bcn_2015_MERGED.csv")

# SPANISH ELECTIONS 20-D 2015 (BARRIS)

# SPANISH ELECTIONS 26-J 2016 (BARRIS)


# CATALAN ELECTIONS 21-D 2017 ( SECCIO CENSAL )

rename_dict = {'JUNTSx CAT (1)': "JxCAT", 'CatComú- Podem (2)': "CECP",
               "C’s":"Cs", "ERC- CatSí":"ERC"}
party_names = ['JxCAT', 'ERC', 'Cs', 'PSC', 'CUP', 'PP', 'CECP']
groups_dict = {'Indep': ['JxCAT', 'ERC', "CUP"],
               'DretD': ['JxCAT', 'ERC', "CUP", "CECP"],
               'Union': ["Cs", "PP", "PSC"]}

# scrape data
data = imp_data_sc_bcn(url_el_cat_bcn_sc_2017)
# process data
df_el_CAT_bcn_SC_2017 = process_imp_data(data, rename_dict, party_names, groups_dict)

# merge population data with city divisions correspondence on DTE, BARRI keys
df_popul_data_MERGED_SC = pd.merge(df_popul_data_MERGED, df_divs, on = ['DTE', 'BARRI'])

# merge population and electoral data
df_el_CAT_bcn_SC_2017_MERGED = pd.merge(df_popul_data_MERGED_SC, df_el_CAT_bcn_SC_2017, on=['DTE', 'SC'])
df_el_CAT_bcn_SC_2017_MERGED["Effective_Educ_univ_pct"] = 100 * (df_el_CAT_bcn_SC_2017_MERGED["Educ_univ"] /
                                                                 df_el_CAT_bcn_SC_2017["Electors"]).round(4)

# save data
df_el_CAT_bcn_SC_2017_MERGED.to_csv("saved_data/el_CAT_bcn_SC_2017_MERGED.csv", index=False)




