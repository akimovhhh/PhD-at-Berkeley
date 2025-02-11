import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Hawaii Market Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import required libraries\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import plotly.express as px\n",
                "from pathlib import Path\n",
                "\n",
                "# Configuration\n",
                "BASE_PATH = Path(\"/Users/akimovh/My documents/projects/predatory pricing/data/DB1B\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Load Reference Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load city market and metropolitan area mappings\n",
                "city_met = pd.read_excel('/Users/akimovh/My documents/projects/predatory pricing/data/dict/city_met.xlsx')\n",
                "city_dict = pd.read_csv(\"/Users/akimovh/My documents/projects/predatory pricing/data/dict/L_CITY_MARKET_ID.csv\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Define Data Processing Functions"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def process_market_data(market_file, city_dict):\n",
                "    \"\"\"Process market data with city information.\"\"\"\n",
                "    market_cols = [\n",
                "        'ItinID', 'TkCarrier', 'OpCarrier', 'MktCoupons', \n",
                "        'OriginCityMarketID', 'DestCityMarketID', 'OriginCountry',\n",
                "        'OriginState', 'DestCountry', 'DestState'\n",
                "    ]\n",
                "    \n",
                "    market_data = pd.read_csv(market_file, usecols=market_cols)\n",
                "    market_data = market_data.query(\"MktCoupons == 1\")\n",
                "    \n",
                "    # Add city names\n",
                "    market_data = pd.merge(market_data, city_dict, left_on='OriginCityMarketID', right_on='Code', how='left')\n",
                "    market_data = pd.merge(market_data, city_dict, left_on='DestCityMarketID', right_on='Code', how='left')\n",
                "    \n",
                "    # Create pairs\n",
                "    market_data['city_pair'] = market_data.apply(\n",
                "        lambda row: str(tuple(sorted([row['Description_x'], row['Description_y']]))), \n",
                "        axis=1\n",
                "    )\n",
                "    market_data['state_pair'] = market_data.apply(\n",
                "        lambda row: str(tuple(sorted([row['OriginState'], row['DestState']]))), \n",
                "        axis=1\n",
                "    )\n",
                "    market_data['country_pair'] = market_data.apply(\n",
                "        lambda row: str(tuple(sorted([row['OriginCountry'], row['DestCountry']]))), \n",
                "        axis=1\n",
                "    )\n",
                "    \n",
                "    return market_data[['ItinID', 'TkCarrier', 'city_pair', 'OpCarrier', 'state_pair', 'country_pair']]\n",
                "\n",
                "def process_ticket_data(ticket_file, itin_ids):\n",
                "    \"\"\"Process ticket data for given itinerary IDs.\"\"\"\n",
                "    ticket_cols = [\n",
                "        'ItinID', 'Coupons', 'RoundTrip', 'DollarCred', 'Year',\n",
                "        'Quarter', 'ItinFare', 'BulkFare', 'Distance', 'DistanceGroup',\n",
                "        'Passengers', 'OnLine'\n",
                "    ]\n",
                "    \n",
                "    tickets = pd.read_csv(ticket_file, usecols=ticket_cols)\n",
                "    mask = (\n",
                "        tickets['ItinID'].isin(itin_ids) & \n",
                "        (tickets['DollarCred'] == 1) & \n",
                "        (tickets['BulkFare'] == 0) & \n",
                "        (tickets['OnLine'] == 1)\n",
                "    )\n",
                "    return tickets[mask]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Load and Process Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize list for storing results\n",
                "all_tickets = []\n",
                "\n",
                "# Process data for each year and quarter\n",
                "for year in range(2017, 2025):\n",
                "    for quarter in range(1, 5):\n",
                "        # Skip 2024 Q4\n",
                "        if year == 2024 and quarter == 4:\n",
                "            continue\n",
                "            \n",
                "        # Define file paths\n",
                "        ticket_file = BASE_PATH / f\"Ticket/{year}_{quarter}.csv\"\n",
                "        market_file = BASE_PATH / f\"Market/Origin_and_Destination_Survey_DB1BMarket_{year}_{quarter}.csv\"\n",
                "        \n",
                "        # Process market data\n",
                "        market_data = process_market_data(market_file, city_dict)\n",
                "        itin_ids = market_data.ItinID.unique()\n",
                "        \n",
                "        # Process ticket data\n",
                "        ticket_data = process_ticket_data(ticket_file, itin_ids)\n",
                "        \n",
                "        # Merge and append\n",
                "        merged = pd.merge(market_data, ticket_data, on='ItinID')\n",
                "        all_tickets.append(merged)\n",
                "\n",
                "# Combine all data\n",
                "tickets = pd.concat(all_tickets)\n",
                "\n",
                "# Add date column\n",
                "tickets['date'] = pd.to_datetime(tickets['Year'].astype(str) + '-' + \n",
                "                               ((tickets['Quarter'] - 1) * 3 + 1).astype(str) + '-01')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Add Flight Classifications"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define and add flight types\n",
                "conditions = [\n",
                "    tickets['state_pair'] == \"('HI', 'HI')\",\n",
                "    tickets['country_pair'] == \"('US', 'US')\",\n",
                "]\n",
                "choices = ['Interstate', 'Other domestic']\n",
                "\n",
                "tickets['flight_type'] = np.select(conditions, choices, default='Bad data')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Save Processed Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Save processed data\n",
                "tickets.to_csv(\"/Users/akimovh/My documents/projects/predatory pricing/data/temporal/HI_flights.csv\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Passenger Volume Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create passenger volume analysis\n",
                "plot_data = tickets.groupby(['date', 'flight_type'], as_index=False).Passengers.sum()\n",
                "plot_data['Passengers'] = plot_data['Passengers'] * 10  # Scale factor\n",
                "\n",
                "# Create and display plot\n",
                "fig = px.line(plot_data, x=\"date\", y=\"Passengers\", color='flight_type',\n",
                "              title='Passenger Volume by Flight Type')\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Market Share Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate market shares\n",
                "plot_data = tickets.groupby(['date', 'flight_type'], as_index=False).Passengers.sum()\n",
                "plot_data['total_Passengers'] = plot_data.groupby('date').Passengers.transform('sum')\n",
                "plot_data['Share'] = plot_data['Passengers'] / plot_data['total_Passengers'] * 100\n",
                "\n",
                "# Create and display plot\n",
                "fig = px.line(plot_data, x=\"date\", y=\"Share\", color='flight_type',\n",
                "              title='Market Share by Flight Type')\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Hawaiian Airlines Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Filter for Hawaiian Airlines\n",
                "hawaiian_data = tickets[tickets['TkCarrier'] == 'HA'].copy()\n",
                "\n",
                "# Add specific flight type classifications for Hawaiian\n",
                "conditions = [\n",
                "    hawaiian_data['state_pair'] == \"('HI', 'HI')\",\n",
                "    hawaiian_data['state_pair'].str.contains(\"'HI'\"),\n",
                "]\n",
                "choices = ['Interstate HI', 'Hi-Mainland']\n",
                "\n",
                "hawaiian_data['flight_type'] = np.select(conditions, choices, default='Other domestic')\n",
                "\n",
                "# Create passenger volume analysis for Hawaiian\n",
                "plot_data = hawaiian_data.query(\"flight_type in ['Interstate HI', 'Hi-Mainland']\")\\\n",
                "    .groupby(['date', 'flight_type'], as_index=False).Passengers.sum()\n",
                "plot_data['Passengers'] = plot_data['Passengers'] * 10\n",
                "\n",
                "# Create and display plot\n",
                "fig = px.line(plot_data, x=\"date\", y=\"Passengers\", color='flight_type',\n",
                "              title='Hawaiian Airlines Passenger Volume by Route Type')\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Hawaiian Airlines Market Share Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate market shares for Hawaiian Airlines routes\n",
                "plot_data = hawaiian_data.query(\"flight_type in ['Interstate HI', 'Hi-Mainland']\")\\\n",
                "    .groupby(['date', 'flight_type'], as_index=False).Passengers.sum()\n",
                "plot_data['total_Passengers'] = plot_data.groupby('date').Passengers.transform('sum')\n",
                "plot_data['Share'] = plot_data['Passengers'] / plot_data['total_Passengers'] * 100\n",
                "\n",
                "# Create and display plot\n",
                "fig = px.line(plot_data, x=\"date\", y=\"Share\", color='flight_type',\n",
                "              title='Hawaiian Airlines Market Share by Route Type')\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Interisland Flight Analysis with Fare Information"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Filter for interisland flights\n",
                "inner_ha = tickets[tickets['state_pair'] == \"('HI', 'HI')\"].query(\"TkCarrier in ['HA', 'WN', 'WP']\")\n",
                "\n",
                "# Create passenger volume analysis by carrier\n",
                "plot_data = inner_ha.groupby(['date', 'TkCarrier'], as_index=False).Passengers.sum()\n",
                "plot_data['Passengers'] = plot_data['Passengers'] * 10\n",
                "\n",
                "# Create and display plot\n",
                "fig = px.line(plot_data, x=\"date\", y=\"Passengers\", color='TkCarrier',\n",
                "              title='Interisland Passenger Volume by Carrier')\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Fare Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Process fare data\n",
                "inner_ha_agg = inner_ha.groupby(['TkCarrier', 'date', 'city_pair'], as_index=False).Passengers.sum()\n",
                "\n",
                "# Filter fares and calculate aggregates\n",
                "inner_ha_fares = inner_ha.query('ItinFare>20')\n",
                "inner_ha_fares['passengers_per_quarter'] = inner_ha_fares.groupby(\n",
                "    ['TkCarrier', 'city_pair', 'date']\n",
                ").Passengers.transform('sum')\n",
                "\n",
                "# Calculate cumulative passengers and filter outliers\n",
                "inner_ha_fares = inner_ha_fares.sort_values(by='ItinFare')\n",
                "inner_ha_fares['passengers_per_quarter_cum'] = inner_ha_fares.groupby(\n",
                "    ['TkCarrier', 'city_pair', 'date']\n",
                ").Passengers.transform('cumsum')\n",
                "\n",
                "mask = (\n",
                "    (inner_ha_fares['passengers_per_quarter_cum']/inner_ha_fares['passengers_per_quarter']>0.01) & \n",
                "    (inner_ha_fares['passengers_per_quarter_cum']/inner_ha_fares['passengers_per_quarter']<0.99) \n",
                ")\n",
                "inner_ha_fares = inner_ha_fares[mask]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Revenue and Price Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate revenue and aggregate by carrier and route\n",
                "inner_ha_fares['Revenue'] = inner_ha_fares['Passengers'] * inner_ha_fares['ItinFare']\n",
                "inner_ha_fares = inner_ha_fares.groupby(\n",
                "    ['TkCarrier', 'date', 'city_pair'], \n",
                "    as_index=False\n",
                ").agg(\n",
                "    Passengers=('Passengers', 'sum'),\n",
                "    Revenue=('Revenue', 'sum')\n",
                ")\n",
                "\n",
                "# Calculate carrier-level aggregates\n",
                "inner_ha_fares_agg = inner_ha_fares.groupby(\n",
                "    ['TkCarrier', 'date'], \n",
                "    as_index=False\n",
                ").agg(\n",
                "    Passengers=('Passengers', 'sum'),\n",
                "    Revenue=('Revenue', 'sum')\n",
                ")\n",
                "\n",
                "# Calculate average prices\n",
                "inner_ha_fares['Price'] = inner_ha_fares.Revenue / inner_ha_fares.Passengers\n",
                "inner_ha_fares_agg['Price'] = inner_ha_fares_agg.Revenue / inner_ha_fares_agg.Passengers"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Price Trend Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create price trend analysis\n",
                "plot_data = inner_ha_fares_agg\n",
                "\n",
                "# Create and display plot\n",
                "fig = px.line(plot_data, x=\"date\", y=\"Price\", color='TkCarrier',\n",
                "              title='Average Fare by Carrier Over Time')\n",
                "fig.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the notebook
with open('hawaii_market_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook saved as 'hawaii_market_analysis.ipynb'")
