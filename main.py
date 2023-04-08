import streamlit as st
import pandas as pd
import pickle as pkl

BIN_PATH = "binaries"

header = st.container()
form_container = st.container()
form = st.form("features")
evaluation = st.container()
output_value = None

data = pd.read_csv("data/telecom_churn.csv")

with header:
    st.image('Customer-Churn.png')
    st.title("Predict Customer Churn")
    st.markdown("As competition and technologies expand in the telecommunication industry, service providers need to "
                "overcome the challenge of keeping existing customers retention. **Customer Churn** happens when a "
                "customer decides to leave the company in search of a better one. The developed model predicts "
                "whether a customer decides to leave or stay with the company given some information.")


def on_submit(day_mins, monthly_charge, cust_serv_calls, contract_renewal):
    customer_data = {
        "ContractRenewal": [contract_renewal],
        "CustServCalls": [cust_serv_calls],
        "DayMins": [day_mins],
        "MonthlyCharge": [monthly_charge]
    }
    customer_data = pd.DataFrame(customer_data)
    print(customer_data)
    with open(f"{BIN_PATH}/model", 'rb') as model_bin:
        model = pkl.load(model_bin)
    with open(f"{BIN_PATH}/sc_x", 'rb') as sc_bin:
        sc_x = pkl.load(sc_bin)
    features = pd.DataFrame(sc_x.transform(customer_data), columns=customer_data.columns)
    output = model.predict(features)
    print("Predicted: ", output)
    return bool(output)


with form_container:
    st.header("Input your customer's information")
    with form:
        st.markdown("Fill in the information below")
        sel_col, disp_col = st.columns(2)
        dm = sel_col.number_input("Average Daytime Minutes", key="DayMins", min_value=0.0, max_value=350.80,
                                  help="Average daytime minutes charged to the customer on a daily basis.")
        mc = sel_col.number_input("Average Monthly Bill ($)", key="MonthlyCharge", min_value=14.0,
                                  max_value=111.30,
                                  help="Customer's average monthly bill.")

        csc = sel_col.slider("Calls to Customer Service", key="CustServCalls", min_value=0, max_value=9,
                             help="Number of times the customer contacted Customer Service")
        cr = int(sel_col.checkbox("Contract Recently Renewed?", key="ContractRenewal",
                                  help="Has your customer recently renewed their contract?"))
        disp_col.image('image2.jpg')
        done = disp_col.form_submit_button("Get Prediction!", use_container_width=True)
        if done:
            output_value = on_submit(day_mins=dm, monthly_charge=mc, cust_serv_calls=csc, contract_renewal=cr)


with evaluation:
    st.header("Will your customer leave the company?")
    img, txt = st.columns([1, 3])
    if output_value is None:
        st.markdown("Fill in the above form to get a prediction!")
    elif output_value:
        img.image('sadjpg.jpg', width=150)
        txt.markdown("**Yes, The customer is likely to leave the company!**")
    else:
        img.image('smile.png', width=150)
        txt.markdown("**No, The customer will likely stay using your services!**")
