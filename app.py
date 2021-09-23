from hydralit import HydraApp
import streamlit as st
from anpr import ANPRApp


if __name__ == "__main__":

    # this is the host application, we add children to it and that's it!
    app = HydraApp(title="Sample Hydralit App", favicon="🐙")

    # add all your application classes here
    app.add_app("Automatic Number Plate Recognition System", icon="🚗", app=ANPRApp())

    # run the whole lot
    app.run()
