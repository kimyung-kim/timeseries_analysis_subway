import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

def detection(df, penalty, name):
    ride_counts = df[['사용일자', '승차총승객수', '하차총승객수']].copy()
    ride_counts['사용일자'] = pd.to_datetime(ride_counts['사용일자'], format='%Y%m%d')
    ride_counts.set_index('사용일자', inplace=True)
    signal = (ride_counts['승차총승객수'] + ride_counts['하차총승객수']).values
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=penalty)
    title = 'Change Point Detection for' + name
    change_point = [cp for cp in result if cp < len(signal)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for cp in change_point:
        ax.axvline(ride_counts.index[cp], color='red', linestyle='--')
    ax.set_title(title)
    ax.plot(ride_counts.index, signal)
    plt.xticks(rotation=45)
    plt.plot(pd.Series(np.array(df['social_distance']*10000), index=ride_counts.index))
    plt.plot(pd.Series(np.array(df['holiday']*10000), index=ride_counts.index))
    plt.show()