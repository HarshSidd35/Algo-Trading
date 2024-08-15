import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


model = load_model('Stock Predictions Model.keras')

st.set_page_config(page_title='Algorithmic Trading And Data Visualization', initial_sidebar_state = 'auto')

st.header('Algorithmic Trading And Data Visualization')
stock =st.text_input('Enter Stock Symnbol :',"GOOG") 
start =  st.date_input("Enter the Staring Date : ") 
end = st.date_input("Enter the Ending Date : ")


def work():

    data = yf.download(stock, start ,end) # downloading data

    st.subheader('Stock Data')
    st.write(data)

    # saperating data 80% training and testing data 20
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    #  MinMax scale in sklearn library
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    st.title("Data Visualization")

    st.subheader("Volumn Chart")
    # Plot the volume chart
    fig0 = plt.figure(figsize=(10, 6))
    plt.bar(data.index, data['Volume'], color='blue', alpha=0.6)
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Volume Chart for ' + stock)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig0)


    # ploting the MA50 and Actual Close price
    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(data.Close, 'g')
    plt.legend(["MA50","Price"],loc="lower right")
    plt.show()
    st.pyplot(fig1)

    # ploting the MA50 VS MA100 and Actual Close price
    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(ma_100_days, 'b')
    plt.plot(data.Close, 'g')
    plt.legend(["MA50","MA100","Price"],loc="lower right")
    plt.show()
    st.pyplot(fig2)

    # ploting the MA100 VS MA200 and Actual Close price
    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_100_days, 'r')
    plt.plot(ma_200_days, 'b')
    plt.plot(data.Close, 'g')

    plt.legend(["MA100","MA200","Price"],loc="lower right")
    plt.show()
    st.pyplot(fig3)

    # ploting the MA50 VS MA100 VS MA200 and Actual Close price
    st.subheader('Price vs MA50 vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days,'g')
    plt.plot(ma_100_days, 'r')
    plt.plot(ma_200_days, 'b')
    plt.plot(data.Close, 'gray')

    plt.legend(["MA50","MA100","MA200","Price"],loc="lower right")
    plt.show()
    st.pyplot(fig3)

    # preadicting price on the basis of test model
    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale
 
    # Plot predicted and origianl price
    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'g', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(["Original Price","Predicted Price"],loc="lower right")
    plt.show()
    st.pyplot(fig4)


st.button(type="primary", label="Submit !", on_click=work)


