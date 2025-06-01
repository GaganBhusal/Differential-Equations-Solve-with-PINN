import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class StreamlitPage():

    def __init__(self):
        super(StreamlitPage, self).__init__()
        self.value_entered = False


    def buildPage(self):
        self.create_left_sidebar()
        self.show_from_sidebar()


    def create_left_sidebar(self):
        self.left_sidebar = st.sidebar.selectbox(
                    "Equations",
                    ("1D Heat Equation", "1D Wave Equation"),
                    index = None,
                    placeholder = "Choose an equation"
                )
        
    def show_from_sidebar(self):
        if self.left_sidebar == "1D Heat Equation":
            self.HeatEquation1D()
        elif self.left_sidebar == "1D Wave Equation":
            self.WaveEquation1D()
        
    def HeatEquation1D(self):

        with st.form("Main"):

            st.write("1D Heat Equation")
            left_column, right_column = st.columns(2)
            with left_column:
                st.markdown("**Length of rod and time**")
                st.write("")
                lcl, lcr = st.columns([1, 1])
                with lcl:
                    st.markdown("**Length of rod**")
                    st.write("")
                    st.markdown("**Total time**")
                with lcr:
                    self.x = st.number_input(
                        "", 
                        value = 1, 
                        placeholder = 1.0,
                        key = "x",
                        label_visibility = "collapsed"
                    )
                    # st.write("")
                    self.t = st.number_input(
                        "", 
                        value = 1, 
                        placeholder = 1.0,
                        key = "t",
                        label_visibility = "collapsed"
                    )
            with right_column:
            
                
                rcl, rcr = st.columns([3, 1])

                with rcl:
                    st.write("**Initial Condition**")
                    st.markdown("**u(x, 0)** ")
                    st.write("**Boundry Condition**")

                    st.markdown("**u(0, t)** ")
                    st.markdown("**u(x, t)** ")

                    
                with rcr:
                    st.write("")
                    st.write("")
                    self.ic_func = st.text_input(
                        "",
                        value = "sin(x)",
                        key = "ic",
                        label_visibility = "collapsed"
                    )

                    st.write("")
                    st.write("")
                    self.bc0 = st.number_input(
                        "",
                        value = 0,
                        placeholder = 0.0,
                        key = "bc0",
                        label_visibility = "collapsed"
                    )

                    self.bc1 = st.number_input(
                        "",
                        value = 0, 
                        placeholder = 0.0,
                        key = "bc1",
                        label_visibility = "collapsed"
                    )
            self.value_entered = st.form_submit_button("Submit")

    def WaveEquation1D(self):
        st.write("1D Wave Equation")


    def getValues(self):
        
        return self.x, self.t, self.ic_func, self.bc0, self.bc1




