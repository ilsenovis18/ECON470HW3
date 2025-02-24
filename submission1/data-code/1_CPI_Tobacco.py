## Title:         CDC Tax Burden on Tobacco
## Author:        Ilse Novis
## Date Created:  2/24/2025
## Date Edited:   3/17/2025
## Description:   Clean and analyze CDC data 


import pandas as pd

# File paths
cig_file = "/Users/ilsenovis/Documents/GitHub/ECON470HW3/data/input/The_Tax_Burden_on_Tobacco__1970-2019.csv"
cpi_file = "/Users/ilsenovis/Documents/GitHub/ECON470HW3/data/input/historical-cpi-u-202501.xlsx"
output_path = "/Users/ilsenovis/Documents/GitHub/ECON470HW3/data/output/"

# Load tobacco data
cig_data = pd.read_csv(cig_file)

# Load CPI data correctly (header is in row 4, so skip first 3 rows)
cpi_data = pd.read_excel(cpi_file, skiprows=3)

# Remove extra spaces in column names
cpi_data.rename(columns=lambda x: str(x).strip(), inplace=True)

# Drop "Indent Level" since it's unnecessary
cpi_data = cpi_data.drop(columns=["Indent Level"], errors="ignore")

# Convert monthly CPI data into long format (melt)
cpi_data = cpi_data.melt(id_vars=["Year"], var_name="month", value_name="index")

# Convert CPI values to numeric (handle errors)
cpi_data["index"] = pd.to_numeric(cpi_data["index"], errors="coerce")

# Compute annual CPI average
cpi_data = cpi_data.groupby("Year", as_index=False)["index"].mean()

# Print to confirm "index" column is correct
print("Processed CPI Data (Should Contain Year and index):")
print(cpi_data.head())

# Convert "Year" column to integer
cpi_data["Year"] = cpi_data["Year"].astype(int)

# Print column names to verify they are correct
print("CPI Data Columns:", cpi_data.columns)

# Clean tobacco data
measure_mapping = {
    "Average Cost per pack": "cost_per_pack",
    "Cigarette Consumption (Pack Sales Per Capita)": "sales_per_capita",
    "Federal and State tax as a Percentage of Retail Price": "tax_percent",
    "Federal and State Tax per pack": "tax_dollar",
    "Gross Cigarette Tax Revenue": "tax_revenue",
    "State Tax per pack": "tax_state"
}

cig_data["measure"] = cig_data["SubMeasureDesc"].map(measure_mapping)
cig_data = cig_data[["LocationAbbr", "LocationDesc", "Year", "Data_Value", "measure"]]
cig_data.rename(columns={"LocationAbbr": "state_abb", "LocationDesc": "state", "Data_Value": "value"}, inplace=True)

# Pivot tobacco data
final_data = cig_data.pivot(index=["state", "Year"], columns="measure", values="value").reset_index()

# Reshape CPI data: Convert monthly values to annual average
cpi_data = cpi_data.melt(id_vars=["Year"], var_name="month", value_name="index")
cpi_data = cpi_data.groupby("Year", as_index=False)["index"].mean()

# Merge CPI data with tobacco data and adjust to 2012 dollars
final_data = final_data.merge(cpi_data, on="Year", how="left")
final_data["price_cpi"] = final_data["cost_per_pack"] * (230 / final_data["index"])
final_data["tax_2012"] = final_data["tax_state"] * (230 / final_data["index"])

# Save final outputs
final_data.to_csv(f"{output_path}TaxBurden_Data.csv", index=False)
final_data.to_pickle(f"{output_path}TaxBurden_Data.pkl")

print(f"Processing complete. Files saved to: {output_path}")



