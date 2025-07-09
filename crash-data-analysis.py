import marimo

__generated_with = "0.14.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Importing data from a CKAN-powered data portal""")
    return


@app.cell
def _():
    import ckanapi
    from pprint import pprint

    site = "https://data.wprdc.org"
    return ckanapi, pprint, site


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define a function to get data from any CKAN resource stored in the CKAN Datastore (an internal database for tabular data).""")
    return


@app.cell
def _(ckanapi):
    def get_resource_data(site,resource_id,count=50):
        # Use the datastore_search API endpoint to get <count> records from
        # a CKAN resource.
        ckan = ckanapi.RemoteCKAN(site)
        response = ckan.action.datastore_search(id=resource_id, limit=count)

        # A typical response is a dictionary like this
        #{u'_links': {u'next': u'/api/action/datastore_search?offset=3',
        #             u'start': u'/api/action/datastore_search'},
        # u'fields': [{u'id': u'_id', u'type': u'int4'},
        #             {u'id': u'pin', u'type': u'text'},
        #             {u'id': u'number', u'type': u'int4'},
        #             {u'id': u'total_amount', u'type': u'float8'}],
        # u'limit': 3,
        # u'records': [{u'_id': 1,
        #               u'number': 11,
        #               u'pin': u'0001B00010000000',
        #               u'total_amount': 13585.47},
        #              {u'_id': 2,
        #               u'number': 2,
        #               u'pin': u'0001C00058000000',
        #               u'total_amount': 7827.64},
        #              {u'_id': 3,
        #               u'number': 1,
        #               u'pin': u'0001C01661006700',
        #               u'total_amount': 3233.59}],
        # u'resource_id': u'd1e80180-5b2e-4dab-8ec3-be621628649e',
        # u'total': 88232}
        data = response['records']
        return data
    return (get_resource_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Get data from the resource with ID "bf8b3c7e-8d60-40df-9134-21606a451c1a" (this is taken from the end of the URL for the 2017 Allegeheny County Crash Data [https://data.wprdc.org/dataset/allegheny-county-crash-data/resource/bf8b3c7e-8d60-40df-9134-21606a451c1a]). Set the row count to get to 999999999, a much larger number than the number of rows in the 2017 crash data.""")
    return


@app.cell
def _(get_resource_data, site):
    crash_data_2017 = get_resource_data(site,resource_id="bf8b3c7e-8d60-40df-9134-21606a451c1a",count=999999999)
    return (crash_data_2017,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""How may rows did we get?""")
    return


@app.cell
def _(crash_data_2017):
    len(crash_data_2017)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Good. We got all of them. What does a sample row look like?""")
    return


@app.cell
def _(crash_data_2017, pprint):
    pprint(crash_data_2017[2])
    return


@app.cell
def _(crash_data_2017):
    sum([c['INJURY_COUNT'] for c in crash_data_2017 if c['INJURY_COUNT'] is not None]) #crash_injuries_2017
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define a function to add up numbers in a field, ignoring None values (blanks).""")
    return


@app.cell
def _(crash_data_2017):
    def sum_over_field(table,field_name):
        return sum([c[field_name] for c in crash_data_2017 if c[field_name] is not None])
    return (sum_over_field,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check that it works as expected:""")
    return


@app.cell
def _(crash_data_2017, sum_over_field):
    sum_over_field(crash_data_2017,'INJURY_COUNT')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""What fraction of crashes involved pedestrians?""")
    return


@app.cell
def _(crash_data_2017, sum_over_field):
    total_2017_crashes = len(crash_data_2017)
    sum_over_field(crash_data_2017,'PEDESTRIAN')/total_2017_crashes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Sorting the boolean indicators by counts""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""There's a lot of different boolean values (those having values of either 0 or 1 like "ALCOHOL_RELATED") describing each event. Let's do a little analysis to try to get a handle on the relative prominence of each of these indicators.""")
    return


@app.cell
def _(crash_data_2017, sum_over_field):
    boolean_fields = ['AGGRESSIVE_DRIVING', 'ALCOHOL_RELATED', 'BICYCLE', 'CELL_PHONE', 'COMM_VEHICLE', 'CROSS_MEDIAN', 
                      'CURVED_ROAD', 'CURVE_DVR_ERROR', 'DEER_RELATED', 'DISTRACTED', 'DRINKING_DRIVER', 'DRIVER_16YR',
                      'DRIVER_17YR', 'DRIVER_18YR', 'DRIVER_19YR', 'DRIVER_20YR', 'DRIVER_50_64YR', 'DRIVER_65_74YR', 'DRIVER_75PLUS',
                      'DRUGGED_DRIVER', 'DRUG_RELATED', 'FATAL', 'FATAL_OR_MAJ_INJ', 'FATIGUE_ASLEEP', 'FIRE_IN_VEHICLE', 
                      'HAZARDOUS_TRUCK', 'HIT_BARRIER', 'HIT_BRIDGE', 'HIT_DEER', 'HIT_EMBANKMENT', 'HIT_FIXED_OBJECT',
                      'HIT_GDRAIL', 'HIT_GDRAIL_END', 'HIT_PARKED_VEHICLE', 'HIT_POLE', 'HIT_TREE_SHRUB', 'HO_OPPDIR_SDSWP',
                      'HVY_TRUCK_RELATED', 'ICY_ROAD', 'ILLEGAL_DRUG_RELATED', 'ILLUMINATION_DARK', 'IMPAIRED_DRIVER', 'INJURY', 
                      'INJURY_OR_FATAL', 'INTERSECTION', 'INTERSTATE', 'LANE_CLOSED', 'LIMIT_65MPH', 'LOCAL_ROAD', 'LOCAL_ROAD_ONLY', 
                      'MAJOR_INJURY', 'MC_DRINKING_DRIVER', 'MINOR_INJURY', 'MODERATE_INJURY', 'MOTORCYCLE', 'NHTSA_AGG_DRIVING', 
                      'NON_INTERSECTION', 'NO_CLEARANCE', 'OVERTURNED', 'PEDESTRIAN', 'PHANTOM_VEHICLE', 'PROPERTY_DAMAGE_ONLY', 
                      'PSP_REPORTED', 'REAR_END', 'RUNNING_RED_LT', 'RUNNING_STOP_SIGN', 'SCHOOL_BUS', 'SCHOOL_ZONE', 
                      'SHLDR_RELATED', 'SIGNALIZED_INT', 'SNOW_SLUSH_ROAD', 'SPEEDING', 'SPEEDING_RELATED', 'STATE_ROAD',
                      'STOP_CONTROLLED_INT', 'SUDDEN_DEER', 'SV_RUN_OFF_RD', 'TAILGATING', 'TRAIN', 'TRAIN_TROLLEY', 
                      'TROLLEY', 'TURNPIKE', 'UNBELTED', 'UNDERAGE_DRNK_DRV', 'UNLICENSED', 'UNSIGNALIZED_INT',
                      'VEHICLE_FAILURE', 'VEHICLE_TOWED', 'WET_ROAD', 'WORK_ZONE']
    boolean_results = {f: sum_over_field(crash_data_2017,f) for f in boolean_fields}
    [(k, boolean_results[k]) for k in sorted(boolean_results, key=boolean_results.get, reverse=True)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Using Pandas dataframes""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Load the data into a Pandas dataframe (kind of a spreadsheet-like data structure) to take advantage of the power of Pandas.""")
    return


@app.cell
def _(crash_data_2017):
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(crash_data_2017)
    return df, np, pd


@app.cell
def _(df, pd):
    pd.options.display.max_columns = None
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Show the records where the vehicle count was 10 (the observed maximum):""")
    return


@app.cell
def _(df):
    df[df.VEHICLE_COUNT == 10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Select just the columns involving vehicle counts to find the distribution of vehicles in this accident:""")
    return


@app.cell
def _(df):
    df[df.VEHICLE_COUNT == 10].loc[:,['AUTOMOBILE_COUNT','BICYCLE_COUNT','BUS_COUNT','COMM_VEH_COUNT','HEAVY_TRUCK_COUNT','MOTORCYCLE_COUNT','SMALL_TRUCK_COUNT','SUV_COUNT','VAN_COUNT','VEHICLE_COUNT']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""One of the two 10-vehicle accidents in 2017 involved 7 SUVs!""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot a histogram of accidents by vehicle count.""")
    return


@app.cell
def _(crash_data_2017):
    import matplotlib.pyplot as plt
    #df.plot.hist(by="VEHICLE_COUNT", bins=11)
    vehicle_counts = [c["VEHICLE_COUNT"] for c in crash_data_2017 if c["VEHICLE_COUNT"] is not None]
    histog = plt.hist(vehicle_counts, density=False, bins=11)
    histog[2]
    return plt, vehicle_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""So one-car accidents are about half as common as two-car accidents, and then crashes with larger vehicle counts seem to drop off rapidly. To examine how they drop off, let's replot the histogram with a logarithmic y-axis:""")
    return


@app.cell
def _(plt, vehicle_counts):
    plt.hist(vehicle_counts, density=False, bins=11, log=True)[2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The straight line that one could draw between *n* = 2 and *n* = 7 suggests that there's roughly an exponential drop-off in the frequency of *n*-vehicle accidents for *n* > 1. The bulge at the end represents a small number of accidents, but there could be an effect where car crashes in heavier traffic tend to avalanche, leading to more vehicles being sucked into the accident than one would expect based on the overall distribution.

    And what about that single crash on the left of the distribution that has a vehicle count of zero? Investigating that is left as an exercise to the reader.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Using SQL Queries""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The WPRDC also provides for your convenience a cumulative table that has records for all Allegheny County vehicle crashes from 2004 through 2017: https://data.wprdc.org/dataset/allegheny-county-crash-data/resource/2c13021f-74a9-4289-a1e5-fe0472c89881""")
    return


@app.cell
def _():
    cumulative_resource_id = "2c13021f-74a9-4289-a1e5-fe0472c89881"
    return (cumulative_resource_id,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    While you can download the entire CSV file if you want using our streaming downloader from [https://tools.wprdc.org/downstream/2c13021f-74a9-4289-a1e5-fe0472c89881](https://tools.wprdc.org/downstream/2c13021f-74a9-4289-a1e5-fe0472c89881),
    you can also use SQL queries to get subsets of the crash records or to get the SQL database to do some of the computation for you.

    First we define a function to run a SQL query on a given CKAN site:
    """
    )
    return


@app.cell
def _(ckanapi):
    def query_resource(site,query):
        # Use the datastore_search_sql API endpoint to query a CKAN resource.
        ckan = ckanapi.RemoteCKAN(site)
        response = ckan.action.datastore_search_sql(sql=query)
        # A typical response is a dictionary like this
        #{u'fields': [{u'id': u'_id', u'type': u'int4'},
        #             {u'id': u'_full_text', u'type': u'tsvector'},
        #             {u'id': u'pin', u'type': u'text'},
        #             {u'id': u'number', u'type': u'int4'},
        #             {u'id': u'total_amount', u'type': u'float8'}],
        # u'records': [{u'_full_text': u"'0001b00010000000':1 '11':2 '13585.47':3",
        #               u'_id': 1,
        #               u'number': 11,
        #               u'pin': u'0001B00010000000',
        #               u'total_amount': 13585.47},
        #              {u'_full_text': u"'0001c00058000000':3 '2':2 '7827.64':1",
        #               u'_id': 2,
        #               u'number': 2,
        #               u'pin': u'0001C00058000000',
        #               u'total_amount': 7827.64},
        #              {u'_full_text': u"'0001c01661006700':3 '1':1 '3233.59':2",
        #               u'_id': 3,
        #               u'number': 1,
        #               u'pin': u'0001C01661006700',
        #               u'total_amount': 3233.59}]
        # u'sql': u'SELECT * FROM "d1e80180-5b2e-4dab-8ec3-be621628649e" LIMIT 3'}
        data = response['records']
        return data
    return (query_resource,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Then as a first example, we can write a query to count up 2017 crash records, grouping them by the month that the crash took place in.""")
    return


@app.cell
def _(query_resource, site):
    crashes_by_month = query_resource(site,
        query='SELECT \"CRASH_MONTH\"::integer as month, count(\"_id\") as count FROM "bf8b3c7e-8d60-40df-9134-21606a451c1a" GROUP BY month ORDER BY month')
    return (crashes_by_month,)


@app.cell
def _(crashes_by_month):
    crashes_by_month
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's import some libraries so we can make a bar chart of 2017 crashes by month.""")
    return


@app.cell
def _():
    import seaborn as sns
    sns.set(style="white", context="talk")
    return (sns,)


@app.cell
def _(crashes_by_month, np, sns):
    x = np.arange(1, 13)
    _y = np.array([int(c['count']) for c in crashes_by_month])
    _by_month = sns.barplot(x=x, y=_y, hue=_y, palette='BuGn_d', legend=False)
    _, _ = (_by_month.set_ylabel('Crashes'), _by_month.set_xlabel('Month'))
    _by_month
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's tweak the previous query a little, using the year instead of the month and switching the resource ID to that for the cumulative (2004-2017) data.""")
    return


@app.cell
def _(cumulative_resource_id, query_resource, site):
    crashes_by_year = query_resource(site,
                        query='SELECT \"CRASH_YEAR\"::integer as year, count(\"_id\")::integer as count FROM "{}" GROUP BY year ORDER BY year'.format(cumulative_resource_id))
    return (crashes_by_year,)


@app.cell
def _(crashes_by_year, np, sns):
    _years = np.array([c['year'] for c in crashes_by_year])
    _y = np.array([c['count'] for c in crashes_by_year])
    _by_month = sns.barplot(x=_y, y=_years, hue=_y, legend=False, palette='Blues_d', orient='h')
    _, _ = (_by_month.set_xlabel('Crashes'), _by_month.set_ylabel('Year'))
    _by_month
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Are there any trends in alcohol-related car crashes?""")
    return


@app.cell
def _(cumulative_resource_id, query_resource, site):
    years_of_alcohol_related_crashes = query_resource(site,
        query=f'SELECT "CRASH_YEAR" as year FROM "{cumulative_resource_id}" WHERE "ALCOHOL_RELATED" = 1 ORDER BY year')
    return (years_of_alcohol_related_crashes,)


@app.cell
def _(years_of_alcohol_related_crashes):
    from collections import Counter
    alcohol_related_crashes_by_year = Counter([d['year'] for d in years_of_alcohol_related_crashes])
    alcohol_related_crashes_by_year
    return (alcohol_related_crashes_by_year,)


@app.cell
def _(alcohol_related_crashes_by_year):
    alcohol_related_crashes_by_year.keys()
    return


@app.cell
def _(alcohol_related_crashes_by_year, np, sns):
    _years2 = np.array(list(alcohol_related_crashes_by_year.keys()))
    _y2 = np.array(list(alcohol_related_crashes_by_year.values()))
    _by_month2 = sns.barplot(x=_y2, y=_years2, hue=_y2, legend=False, palette='Reds_d', orient='h')
    _, _ = (_by_month2.set_xlabel('Alcohol-related crashes'), _by_month2.set_ylabel('Year'))
    _by_month2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""There might be a decrease in the absolute number of alcohol-related crashes from the 2004-2010 period to the 2011-2017 period, but it's difficult to say. (As of 2025, the trend looks clearer, but the reduction may simply be a reduction in overall number of crashes.) We can try normalizing alcohol-related crashes by year with respect to total crashes per year:""")
    return


@app.cell
def _(alcohol_related_crashes_by_year, crashes_by_year, np, sns):
    alcohol_related_counts = list(alcohol_related_crashes_by_year.values())
    all_counts = [c['count'] for c in crashes_by_year]
    ratios = [al / c for al, c in zip(alcohol_related_counts, all_counts)]
    _years3 = sorted(list(alcohol_related_crashes_by_year.keys()))
    _y3 = np.array(ratios)
    _by_month3 = sns.barplot(x=_y3, y=_years3, hue=_y3, legend=False, palette='YlOrRd_r', orient='h')
    _, _ = (_by_month3.set_xlabel('Ratio of alcohol-related crashes'), _by_month3.set_ylabel('Year'))
    _by_month3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Again, there could be a decreasing trend (with noise on top of it), but this is hardly conclusive. (But now, as of 2025, the trend does appear more convincing.)""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
