import streamlit as st
import numpy as np
import torch
import torch.nn as nn

from HeatEquation1D import StreamlitPage
from heat import PINN


class mainPage(StreamlitPage):

    def __init__(self):
        super().__init__()
        

    def start_pipeline(self):
        self.building_pages()
        self.start_training()


    def building_pages(self):

        self.page = StreamlitPage()
        self.page.buildPage()

    
    def start_training(self):
        
        if self.page.value_entered:
            
            self.x, self.t, self.ic0, self.bc0, self.bc1 = self.page.getValues()
            self.HE1D = PINN(self.x, self.t, self.ic0, self.bc0, self.bc1)
            self.page.value_entered = False
            self.HE1D.train()

            st.pyplot(self.HE1D.plot_test_3d())
            st.pyplot(self.HE1D.plot_loss())
            



main = mainPage()
main.start_pipeline()