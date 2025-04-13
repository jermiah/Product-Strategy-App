import pandas as pd
import numpy as np

def compute_product_kpis(main_df):
    """
    Computes key performance indicators (KPIs) for each product, including:
      - Basic KPIs (quantity, revenue, customers)
      - Engagement metrics (repeat purchases, share of wallet)
      - Promotional activity
      - Purchase frequency analysis
      - Day/time trends
    """
    
    # ---------------------------------------------------------------------
    # 1) Data Preparation & Validations
    # ---------------------------------------------------------------------
    main_df.columns = main_df.columns.str.strip()  # Clean column names
    main_df["DT_VENTE"] = pd.to_datetime(main_df["DT_VENTE"])  # Ensure correct datetime format
    
    # Required columns
    required_cols = [
        "CD_ARTICLE", "DT_VENTE", "QT_UVC", "MT_TTC_NET", "PAYMENT_TYPE",
        "day_category", "time_of_day", "CUSTOMER_ID", "CITY_NAME",
        "IS_PROMO_NATIONALE", "IS_PROMO_MAGASIN"
    ]
    
    lb_columns = [col for col in main_df.columns if col.startswith("LB_")]
    required_cols.extend(lb_columns)
    
    for col in required_cols:
        if col not in main_df.columns:
            raise KeyError(f"Column '{col}' is missing from the input DataFrame.")
    
    # Latest transaction date
    latest_date = main_df["DT_VENTE"].max()
    
    # ---------------------------------------------------------------------
    # 2) Basic Product-Level KPIs
    # ---------------------------------------------------------------------
    kpi_df = main_df.groupby("CD_ARTICLE").agg(
        Count_of_Purchases=("CD_ARTICLE", "count"),
        Total_Quantity=("QT_UVC", "sum"),
        Total_Revenue=("MT_TTC_NET", "sum"),
        Avg_Quantity=("QT_UVC", "mean"),
        Avg_Revenue=("MT_TTC_NET", "mean"),
        Last_Purchase_Date=("DT_VENTE", "max"),
        Number_of_Customers=("CUSTOMER_ID", "nunique"),  # Use correct name directly
        Promotion_Nationale_Count=("IS_PROMO_NATIONALE", "sum"),
        Promotion_Magasin_Count=("IS_PROMO_MAGASIN", "sum")
    ).reset_index()
    
    # Compute recency in days
    kpi_df["Days_Since_Last_Purchase"] = (latest_date - kpi_df["Last_Purchase_Date"]).dt.days
    
    # ---------------------------------------------------------------------
        # 3) Repeat Purchase Metrics
        # ---------------------------------------------------------------------
    product_customer_df = (
        main_df.groupby(["CD_ARTICLE", "CUSTOMER_ID"])["CD_ARTICLE"]
        .count()
        .rename("times_purchased")
        .reset_index()
    )
    
    repeat_stats = (
        product_customer_df
        .groupby("CD_ARTICLE")["times_purchased"]
        .agg(
            Repeat_Customers=lambda x: (x > 1).sum(),
            Avg_Purchase_Frequency="mean"
        )
        .reset_index()
    )
    
    # Merge repeat purchase stats
    kpi_df = kpi_df.merge(repeat_stats, on="CD_ARTICLE", how="left")
    kpi_df["Repeat_Purchase_Rate"] = (kpi_df["Repeat_Customers"] / kpi_df["Number_of_Customers"]).fillna(0)
    
    # ---------------------------------------------------------------------
    # 4) Average Share of Wallet (Loyalty)
    # ---------------------------------------------------------------------
    customer_spend = main_df.groupby("CUSTOMER_ID")["MT_TTC_NET"].sum().rename("customer_total_spend").reset_index()
    merged_df = main_df.merge(customer_spend, on="CUSTOMER_ID", how="left")
    
    product_customer_spend = (
        merged_df.groupby(["CD_ARTICLE", "CUSTOMER_ID"])
        .agg(
            product_spend=("MT_TTC_NET", "sum"),
            customer_total_spend=("customer_total_spend", "first")
        )
        .reset_index()
    )
    
    product_customer_spend["fraction_of_customer_spend"] = (
        product_customer_spend["product_spend"] / product_customer_spend["customer_total_spend"]
    )
    
    sow_df = (
        product_customer_spend
        .groupby("CD_ARTICLE")["fraction_of_customer_spend"]
        .mean()
        .rename("Average_Share_Of_Wallet")
        .reset_index()
    )
    
    kpi_df = kpi_df.merge(sow_df, on="CD_ARTICLE", how="left")
        
    # LB_* Hierarchy
    lb_hierarchy = main_df.groupby("CD_ARTICLE")[lb_columns].first().reset_index()
    # City-wise counts
    city_freq = main_df.groupby(["CD_ARTICLE", "CITY_NAME"]).size().unstack(fill_value=0).reset_index()
    
    # Payment type counts
    payment_counts = main_df.groupby(["CD_ARTICLE", "PAYMENT_TYPE"]).size().unstack(fill_value=0).reset_index()
    most_used_payment = main_df.groupby("CD_ARTICLE")["PAYMENT_TYPE"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else None
    ).reset_index().rename(columns={"PAYMENT_TYPE": "most_used_payment_type"})
    
    # Compute the most purchased day category (mode of day_category) for each product
    most_purchased_day_category = main_df.groupby("CD_ARTICLE")["day_category"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else None
    ).reset_index().rename(columns={"day_category": "Most_Purchased_Day_Category"})
    
    # Compute the most purchased time of day (mode of time_of_day) for each product
    most_purchased_time_of_day = main_df.groupby("CD_ARTICLE")["time_of_day"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else None
    ).reset_index().rename(columns={"time_of_day": "Most_Purchased_Time_Of_Day"})
    
    kpi_df = kpi_df.merge(most_purchased_day_category, on="CD_ARTICLE", how="left")
    kpi_df = kpi_df.merge(most_purchased_time_of_day, on="CD_ARTICLE", how="left")
    kpi_df = kpi_df.merge(lb_hierarchy, on="CD_ARTICLE", how="left")
    kpi_df = kpi_df.merge(city_freq, on="CD_ARTICLE", how="left")
    kpi_df = kpi_df.merge(payment_counts, on="CD_ARTICLE", how="left")
    kpi_df = kpi_df.merge(most_used_payment, on="CD_ARTICLE", how="left")
    # ---------------------------------------------------------------------
    # 6) Day & Time Trends
    # ---------------------------------------------------------------------
    day_freq = main_df.groupby(["CD_ARTICLE", "day_category"]).size().unstack(fill_value=0).reset_index()
    time_freq = main_df.groupby(["CD_ARTICLE", "time_of_day"]).size().unstack(fill_value=0).reset_index()
    
    kpi_df = kpi_df.merge(day_freq, on="CD_ARTICLE", how="left")
    kpi_df = kpi_df.merge(time_freq, on="CD_ARTICLE", how="left")
    
    for col in lb_columns:
        kpi_df.rename(columns={col: col.replace("LB_", "Hierarchy_")}, inplace=True)
    
    # Rename newly added columns for city frequency
    for col in city_freq.columns[1:]:  # Exclude 'CD_ARTICLE'
        kpi_df.rename(columns={col: f"City_{col}_Purchases"}, inplace=True)
    
    kpi_df.rename(columns={
        "promo_nationale_count": "Promotion_Nationale_Count",
        "promo_magasin_count": "Promotion_Magasin_Count"
    }, inplace=True)
    
    # Rename payment count columns if present (e.g., CARD, CASH)
    if "CARD" in kpi_df.columns:
        kpi_df.rename(columns={"CARD": "Card_Purchase_Count"}, inplace=True)
    if "CASH" in kpi_df.columns:
        kpi_df.rename(columns={"CASH": "Cash_Purchase_Count"}, inplace=True)
    
    # Rename frequency columns for day_category (e.g., Weekday, Weekend)
    for col in day_freq.columns[1:]:
        kpi_df.rename(columns={col: f"{col}_Purchases"}, inplace=True)
    
    # Rename frequency columns for time_of_day (e.g., Morning, Afternoon, Evening)
    for col in time_freq.columns[1:]:
        kpi_df.rename(columns={col: f"{col}_Purchases"}, inplace=True)
    
    kpi_df.rename(columns={
        "CD_ARTICLE": "Product_Code",
        "Hierarchy_ARTICLE": "Product_Name"
    }, inplace=True)
    
    # ---------------------------------------------------------------------
    # 7) Final Column Ordering
    # ---------------------------------------------------------------------
    
    fixed_cols = [
        "Product_Code", "Product_Name", "Count_of_Purchases", "Total_Quantity", "Total_Revenue",
        "Avg_Quantity", "Avg_Revenue", "Number_of_Customers",
        "Repeat_Customers", "Repeat_Purchase_Rate", "Avg_Purchase_Frequency",
        "Average_Share_Of_Wallet",
        "Last_Purchase_Date", "Days_Since_Last_Purchase"
    ]
    
    other_cols = [col for col in kpi_df.columns if col not in fixed_cols]
    kpi_df = kpi_df[fixed_cols + other_cols]
    
    return kpi_df
