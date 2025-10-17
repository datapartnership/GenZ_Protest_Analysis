import matplotlib.pyplot as plt
import pandas as pd
from wbpyplot import wb_plot


def pretty_indicator(indicator_name):
    """Convert indicator name to a prettier format for display"""
    return indicator_name.replace('_', ' ')

def plot_indicator_groups_wb(indicator_name: str, df_data: pd.DataFrame, window_n: int = 5, sort_df = None):
    # Sort countries by protest counts and group them
    available_countries = df_data['country_name'].unique()
    
    # Filter mena_country_counts to only include countries in our dataset
    filtered_protest_counts = sort_df[
        sort_df['country'].isin(available_countries)
    ].copy()
    
    # Create groups of 5 countries each, ordered by protest count
    groups = []
    for i in range(0, len(filtered_protest_counts), 5):
        group = filtered_protest_counts.iloc[i:i+5]['country'].tolist()
        if group:  # Only add non-empty groups
            groups.append(group)
    
    # Create a colored title with the indicator name highlighted
    pretty_name = pretty_indicator(indicator_name)
    title = f"$\\bf{{{pretty_name}}}$ over time by protest frequency"
    subtitle = f"Countries grouped by protest count (highest to lowest)"

    @wb_plot(
        title=title,
        subtitle=subtitle,
        note=[("Source:", "World Bank Social Contract Factsheet & ACLED protest data until September 30, 2025")],
        # palette="wb_categorical",  # optional
    )
    def _plot(axs):
        # Use the figure created by the decorator and rebuild a 2x2 grid
        import numpy as np
        if isinstance(axs, (list, tuple, np.ndarray)) and len(axs) > 0:
            fig = axs[0].figure
        else:
            fig = plt.gcf()
        fig.clf()
        fig.set_size_inches(14, 10)
        axes = fig.subplots(2, 2).flatten()

        panel_titles = ["First Quartile Protests", "Second Quartile Protests", "Third Quartile Protests", "Fourth Quartile Protests"]

        for idx in range(4):
            ax = axes[idx]
            group_countries = groups[idx] if idx < len(groups) else []

            if not group_countries:
                ax.text(0.5, 0.5, "No countries in group", ha='center', va='center', alpha=0.6)
                ax.set_axis_off()
                continue

            has_data = False
            for country in group_countries:
                subset = df_data[df_data['country_name'] == country].sort_values('year')
                subset = subset.dropna(subset=[indicator_name])
                if subset.empty:
                    continue
                
                # Get protest count for legend
                protest_count = filtered_protest_counts[
                    filtered_protest_counts['country'] == country
                ]['protest_count'].iloc[0] if len(filtered_protest_counts[
                    filtered_protest_counts['country'] == country
                ]) > 0 else 0
                
                label = f"{country} ({protest_count:,})"
                ax.plot(subset['year'], subset[indicator_name], marker='o', linewidth=2, label=label)
                has_data = True

            ax.set_title(panel_titles[idx])
            ax.set_ylabel(pretty_indicator(indicator_name))
            
            # Only show legend if there are actual data lines plotted
            if has_data:
                ax.legend(fontsize=8)

        plt.tight_layout()

    _plot()


# New function: plot_country_indicators_wb
def plot_country_indicators_wb(country_name: str, df_data: pd.DataFrame, indicators: list, window_n: int = 5):
    """
    Plot all specified indicators for a single country over time using wbpyplot.
    Args:
        country_name (str): The country to plot.
        df_data (pd.DataFrame): DataFrame with columns ['country_name', 'year', ...indicators].
        indicators (list): List of indicator column names to plot.
        window_n (int): Rolling window for smoothing (optional, default 5).
    """
    import numpy as np
    # Filter data for the selected country
    country_df = df_data[df_data['country_name'] == country_name].sort_values('year')
    if country_df.empty:
        print(f"No data for {country_name}")
        return

    title = f"Indicators for $\\bf{{{country_name}}}$ over time"
    subtitle = f"Social contract indicators (smoothed, window={window_n})"

    @wb_plot(
        title=title,
        subtitle=subtitle,
        note=[("Source:", "World Bank Social Contract Factsheet")],
    )
    def _plot(ax):
        fig = ax.figure
        fig.set_size_inches(12, 7)
        has_data = False
        for ind in indicators:
            if ind not in country_df.columns:
                continue
            series = country_df[["year", ind]].dropna()
            if series.empty:
                continue
            # Optionally smooth with rolling window
            y = series[ind].rolling(window=window_n, min_periods=1, center=True).mean()
            ax.plot(series["year"], y, marker="o", linewidth=2, label=pretty_indicator(ind))
            has_data = True
        ax.set_ylabel("Indicator Value")
        ax.set_xlabel("Year")
        if has_data:
            ax.legend(title="Indicator", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No data for selected indicators", ha='center', va='center', alpha=0.6)
            ax.set_axis_off()
        plt.tight_layout()

    _plot()


def plot_selected_lines_wb(indicator_name: str, df_data: pd.DataFrame, countries: list, window_n: int = 5, sort_df = None):
    """
    Plot selected countries for a given indicator using the same style as plot_indicator_groups_wb
    but in a single panel.
    
    Args:
        indicator_name (str): Name of the indicator to plot
        df_data (pd.DataFrame): DataFrame containing the data
        countries (list): List of country names to plot
        window_n (int): Rolling window size for smoothing
        sort_df (pd.DataFrame): Optional dataframe with protest counts for legend
    """
    # Create a colored title with the indicator name highlighted
    pretty_name = pretty_indicator(indicator_name)
    title = f"$\\bf{{{pretty_name}}}$ over time"
    subtitle = "Selected countries comparison"

    @wb_plot(
        title=title,
        subtitle=subtitle,
        note=[("Source:", "World Bank Social Contract Factsheet & ACLED protest data until September 30, 2025")],
    )
    def _plot(axs):
        if isinstance(axs, (list, tuple, np.ndarray)):
            ax = axs[0]
        else:
            ax = axs
            
        fig = ax.figure
        fig.set_size_inches(12, 7)
        has_data = False

        for country in countries:
            subset = df_data[df_data['country_name'] == country].sort_values('year')
            subset = subset.dropna(subset=[indicator_name])
            if subset.empty:
                continue
            
            # Get protest count for legend if available
            if sort_df is not None:
                protest_count = sort_df[
                    sort_df['country'] == country
                ]['protest_count'].iloc[0] if len(sort_df[
                    sort_df['country'] == country
                ]) > 0 else 0
                label = f"{country} ({protest_count:,})"
            else:
                label = country
            
            ax.plot(subset['year'], subset[indicator_name], marker='o', linewidth=2, label=label)
            has_data = True

        ax.set_ylabel(pretty_indicator(indicator_name))
        
        # Set fixed y-axis limits from -1 to 2
        plt.ylim(-1, 2)
        
        # Set up proper y-axis ticks and format
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        ax.yaxis.set_major_locator(MultipleLocator(0.5))  # Set major ticks every 0.5
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Show 2 decimal places
        
        # Add grid lines for better readability
        ax.grid(True, linestyle='--', alpha=0.2)
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Only show legend if there are actual data lines plotted
        if has_data:
            ax.legend(fontsize=9)
        
        plt.tight_layout()

    _plot()

    # General-purpose plotting function using wbpyplot
def plot_lines_wb(
    df_data: pd.DataFrame,
    line_col: str,
    x_col: str,
    value_col: str,
    filter_dict: dict = None,
    window_n: int = 5,
    title: str = None,
    subtitle: str = None,
    note: str = "World Bank Social Contract Factsheet",
    highlight_line: str = None
):
    """
    Plot multiple lines for any column (e.g., indicators or countries) with flexible filtering and x-axis.
    Args:
        df_data (pd.DataFrame): DataFrame containing the data.
        line_col (str): Column to plot as separate lines (e.g., 'country_name' or indicator name).
        x_col (str): Column for x-axis (e.g., 'year').
        value_col (str): Column for y-axis values.
        filter_dict (dict): Optional. Dict of {col: value} to filter the data before plotting.
        window_n (int): Rolling window for smoothing (default 5).
        title (str): Plot title.
        subtitle (str): Plot subtitle.
        note (str): Source note for plot.
        highlight_line (str): Optional. Name of the line to highlight (others will be greyed out).
    """
    import numpy as np
    plot_df = df_data.copy()
    if filter_dict:
        for k, v in filter_dict.items():
            plot_df = plot_df[plot_df[k] == v]
    if plot_df.empty:
        print("No data after filtering.")
        return

    if not title:
        title = f"{value_col} over {x_col} by {line_col}"
    if not subtitle:
        subtitle = f"Filtered by: {filter_dict}" if filter_dict else "All data"
    
    # Make the first word in the title bold
    if title:
        words = title.split(' ', 1)  # Split into first word and rest
        if len(words) > 1:
            title = f"$\\bf{{{words[0]}}}$ {words[1]}"
        else:
            title = f"$\\bf{{{words[0]}}}$"

    @wb_plot(
        title=title,
        subtitle=subtitle,
        note=[("Source:", note)],
    )
    def _plot(axs):
        # Use the same approach as plot_indicator_groups_wb - clear and recreate axes
        import numpy as np
        if isinstance(axs, (list, tuple, np.ndarray)) and len(axs) > 0:
            fig = axs[0].figure
        else:
            fig = plt.gcf()
        fig.clf()
        fig.set_size_inches(12, 7)
        ax = fig.subplots(1, 1)  # Create a single subplot

        has_data = False
        all_values = []
        greyed_lines = []  # Store info for text labels
        
        for key, group in plot_df.groupby(line_col):
            series = group[[x_col, value_col]].dropna()
            if series.empty:
                continue
            
            # Collect all y values for dynamic y-axis limits
            all_values.extend(series[value_col].tolist())
            
            # Determine line styling based on highlight_line parameter
            if highlight_line is not None:
                if str(key) == str(highlight_line):
                    # Highlighted line: bold and colored
                    ax.plot(series[x_col], series[value_col], marker="o", linewidth=3, 
                           label=str(key), alpha=1.0, zorder=10)
                else:
                    # Greyed out lines: thin and transparent
                    ax.plot(series[x_col], series[value_col], marker="o", linewidth=1.5, 
                           label=str(key), alpha=0.3, color='gray', zorder=1)
                    
                    # Store info for text labeling greyed lines
                    if len(series) > 0:
                        # Find the highest and lowest points
                        max_idx = series[value_col].idxmax()
                        min_idx = series[value_col].idxmin()
                        max_y = series[value_col].iloc[series[value_col].values.argmax()]
                        min_y = series[value_col].iloc[series[value_col].values.argmin()]
                        
                        # Use the more extreme point (furthest from zero)
                        if abs(max_y) >= abs(min_y):
                            label_x = series.loc[max_idx, x_col]
                            label_y = max_y
                            point_type = 'max'
                        else:
                            label_x = series.loc[min_idx, x_col]
                            label_y = min_y
                            point_type = 'min'
                        
                        greyed_lines.append({
                            'name': str(key), 
                            'x': label_x, 
                            'y': label_y, 
                            'type': point_type
                        })
            else:
                # Normal plotting when no highlight is specified
                ax.plot(series[x_col], series[value_col], marker="o", linewidth=2, label=str(key))
            
            has_data = True

        ax.set_ylabel(value_col)
        ax.set_xlabel(x_col)
        
        if has_data:
            # Set dynamic y-axis limits based on actual data values
            import numpy as np
            y_min = np.min(all_values)
            y_max = np.max(all_values)
            y_range = y_max - y_min
            
            # Add 10% padding to top and bottom
            padding = y_range * 0.1 if y_range > 0 else 0.1
            ax.set_ylim(y_min - padding, y_max + padding)
            
            # Set decimal formatting for y-axis (since we cleared the figure, wb_plot formatting is bypassed)
            from matplotlib.ticker import MultipleLocator, FormatStrFormatter
            # Dynamic tick spacing based on the range
            if y_range <= 1:
                tick_spacing = 0.2
            elif y_range <= 2:
                tick_spacing = 0.5
            elif y_range <= 5:
                tick_spacing = 1.0
            else:
                tick_spacing = max(1, round(y_range / 6))
            
            ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Show 2 decimal places
            
            # Set integer formatting for x-axis (years should be whole numbers)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))  # Show years as integers
            
            # Add grid lines for better readability
            ax.grid(True, linestyle='--', alpha=0.2)
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add text labels for greyed out lines when highlighting
            if highlight_line is not None and greyed_lines:
                # Sort by y-value to help with positioning
                greyed_lines.sort(key=lambda x: x['y'])
                
                # Position labels to avoid overlap
                for i, line_info in enumerate(greyed_lines):
                    # Determine vertical alignment based on whether it's at max or min point
                    if line_info['type'] == 'max':
                        # Label above the max point
                        v_align = 'bottom'
                        y_offset = y_range * 0.02  # Small offset above
                    else:
                        # Label below the min point
                        v_align = 'top'
                        y_offset = -y_range * 0.02  # Small offset below
                    
                    # Add small horizontal offset to reduce overlap
                    h_offset = (i - len(greyed_lines)/2) * (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
                    
                    label_y = line_info['y'] + y_offset
                    label_x = line_info['x'] + h_offset
                    
                    ax.text(label_x, label_y, line_info['name'], 
                           fontsize=8, alpha=0.7, color='gray',
                           verticalalignment=v_align, horizontalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                   edgecolor='none', alpha=0.8))
            
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "No data for selected lines", ha='center', va='center', alpha=0.6)
            ax.set_axis_off()
        
        plt.tight_layout()

    _plot()