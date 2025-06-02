import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

import numpy as np
import mpld3
import torch
import torch.nn as nn

from webPageClass import StreamlitPage
from HeatEquation1D import PINN


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
            self.plot_3d_chart()


    def plot_3d_chart(self):

        X, Y, Z = self.HE1D.plot_test_3d()
        surface = go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale='Viridis',
            opacity=0.97,
            lighting=dict(ambient=0.6, diffuse=0.95, roughness=0.25, specular=0.1, fresnel=0.1),
            lightposition=dict(x=80, y=120, z=40),
            contours=dict(
                z=dict(show=True, usecolormap = True, highlight = False, project_z = True),
                x=dict(show=False, ),
                y=dict(show=False, )
            ),
            showscale=True
        )

        camera = dict(
            eye=dict(x=0.1, y=2.0, z=0.9),
            up=dict(x=0, y=0, z=0),
            center=dict(x=0, y=0, z=0)
        )

        layout = go.Layout(
            title=dict(
                text="1D Heat Equation",
                x=0.2,
                font=dict(size=22, family='Arial Black', color = 'black')
            ),
            scene=dict(
                xaxis_title='Length',
                yaxis_title='Time',
                zaxis_title='Temperature',
                xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.4)', backgroundcolor="rgba(250,250,250,1)"),
                yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.4)', backgroundcolor="rgba(250,250,250,1)"),
                zaxis=dict(showgrid=True, gridcolor='rgba(220,220,220,0.3)', backgroundcolor="rgba(255,255,255,1)"),
                camera=camera,
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            margin=dict(l=10, r=10, b=10, t=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(245,245,245,0.6)",
            font=dict(family="Segoe UI", size=13, color="#000000")
        )

        fig = go.Figure(data=[surface], layout=layout)
        st.plotly_chart(fig, use_container_width=True)


main = mainPage()
main.start_pipeline()