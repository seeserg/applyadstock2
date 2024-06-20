import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Function to apply adstock
def apply_adstock(df, column, decay_rate):
    adstocked_values = df[column].ewm(alpha=decay_rate).mean()
    return adstocked_values

# Function to format currency
def currency(x, pos):
    return '${:,.0f}'.format(x)

# Streamlit app
st.title('MMM Results Adstock Adjustment Tool')

# Margin slider
fixed_margin = st.sidebar.slider('Set the margin', 0.0, 1.0, 0.35, 0.01)

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)
    
    # Handle missing values
    st.sidebar.header("Handle Missing Values")
    missing_value_option = st.sidebar.selectbox(
        "Choose how to handle missing values",
        ("Fill with 0", "Fill with mean", "Drop rows with missing values")
    )
    
    if missing_value_option == "Fill with 0":
        df = df.fillna(0)
    elif missing_value_option == "Fill with mean":
        df = df.fillna(df.mean())
    elif missing_value_option == "Drop rows with missing values":
        df = df.dropna()
    columns_to_drop = ['% Revenue', 'iROAS', 'iCPO', 'CPM/CPC']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    # Convert necessary columns to numeric types
    numeric_columns = ['Spend', 'Revenue', 'Profit', 'Orders', 'Platform Impressions/Clicks']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Display the data
    st.write("Data Preview:")
    st.write(df.head())
    
    # Get unique channels
    channels = df['Platform Channel'].unique()

    # Advanced Adstock Modeling using SARIMAX
    st.sidebar.header("SARIMAX-based Adstock Rate Adjustments to spend and profit calculations")
    adstock_rates = {}
    for channel in channels:
        mask = df['Platform Channel'] == channel
        if df[mask]['Spend'].sum() == 0:
            # Default adstock rate for channels based on impressions
            new_adstock_rate = 0.5
        elif df[mask]['Spend'].nunique() > 1:
            model = SARIMAX(df[mask]['Revenue'], exog=df[mask][['Spend']], order=(1,0,0))
            results = model.fit(disp=False)
            coeff = results.params['Spend']
            new_adstock_rate = max(0.01, min(1, coeff / 100))  # Ensure the adstock rate is within (0, 1]
        else:
            # For channels with sparse spend data, use a default or calculated decay rate
            new_adstock_rate = 0.5  # Default to a more reasonable rate if only a single spend value is present
        adstock_rates[channel] = new_adstock_rate
        st.sidebar.slider(f"Adstock rate for {channel}", 0.0, 1.0, new_adstock_rate, 0.01)
    
    # Apply adstock at the day level
    for channel in channels:
        mask = df['Platform Channel'] == channel
        df.loc[mask, 'Adstocked Spend'] = apply_adstock(df[mask], 'Spend', adstock_rates[channel])
    
    # Normalize adstocked spend to match total spend
    for channel in channels:
        mask = df['Platform Channel'] == channel
        if df[mask]['Spend'].sum() != 0:  # Skip normalization for channels with zero spend
            total_original_spend = df[mask]['Spend'].sum()
            total_adstocked_spend = df[mask]['Adstocked Spend'].sum()
            normalization_factor = total_original_spend / total_adstocked_spend
            df.loc[mask, 'Adstocked Spend'] *= normalization_factor
    
    # Recalculate metrics at the day level
    df['Adstocked Profit'] = df['Revenue'] * fixed_margin - df['Adstocked Spend']
    df['Profit'] = df['Revenue'] * fixed_margin - df['Spend']
    df['ROAS'] = df['Revenue'] / df['Spend']

    # Dropdown to filter channels in the output
    selected_channels = st.multiselect('Select channels to display', channels, default=channels)
    
    # Filter dataframe based on selected channels
    filtered_df = df[df['Platform Channel'].isin(selected_channels)]
    
    # Display filtered data
    st.write("Filtered Data:")
    st.write(filtered_df.head())

    # Column filter select control
    defaultCols = ['Platform Channel','Day','Spend','Orders','Revenue','Profit','Adstocked Spend','Adstocked Profit']
    columns_to_display = st.multiselect('Select columns to display', filtered_df.columns.tolist(), default=defaultCols)

    # Display summary data
    st.write("Summary Data (Day Grain):")
    st.write(filtered_df[columns_to_display])

    # Download updated data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download updated data as CSV",
        data=csv,
        file_name='updated_media_mix.csv',
        mime='text/csv',
    )

    # Button to generate profit charts
    if st.button('Generate Profit Charts'):
        # Plotting Before and After Profit for each selected channel by date
        st.header("Before and After Profit Comparison by Date")

        for channel in selected_channels:
            channel_df = filtered_df[filtered_df['Platform Channel'] == channel]
            if not channel_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(channel_df['Day'], channel_df['Profit'], label='Before Adstock')
                ax.plot(channel_df['Day'], channel_df['Adstocked Profit'], label='After Adstock')
                ax.set_title(f'Profit Comparison by Date for {channel}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Profit ($)')
                ax.yaxis.set_major_formatter(FuncFormatter(currency))
                ax.legend()
                st.pyplot(fig)
        
    # Button to generate adstock charts
    if st.button("Generate Adstock Charts"):
        st.header("Adstock Curve for Each Channel")

        for channel in selected_channels:
            channel_df = filtered_df[filtered_df['Platform Channel'] == channel]
            if not channel_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(channel_df['Day'], channel_df['Spend'], label='Original Spend')
                ax.plot(channel_df['Day'], channel_df['Adstocked Spend'], label='Adstocked Spend')
                ax.set_title(f'Adstock Curve by Date for {channel}')
                ax.set_xlabel('Day')
                ax.set_ylabel('Spend ($)')
                ax.yaxis.set_major_formatter(FuncFormatter(currency))
                ax.legend()
                st.pyplot(fig)
    
    # Button to display adstock decay rates
    if st.button("Display Adstock Decay Rates"):
        st.header("Adstock Decay Rates for Each Channel")
        
        # Plot the decay rates
        fig, ax = plt.subplots(figsize=(10, 6))
        channels = list(adstock_rates.keys())
        rates = list(adstock_rates.values())
        ax.barh(channels, rates, color='skyblue')
        ax.set_xlabel('Adstock Decay Rate')
        ax.set_title('Adstock Decay Rates by Channel')
        for index, value in enumerate(rates):
            ax.text(value, index, f'{value:.2f}')
        st.pyplot(fig)
