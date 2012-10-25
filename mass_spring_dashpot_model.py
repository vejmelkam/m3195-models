# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:07:22 2012

@author: martin

This code creates, simulates and animates a free vibrations system (spring,
dashpot, m) in a car unicycle model.  Build from example lorenz_ui.py
"""
# Author: Martin Vejmelka <vejmelkam@gmail.com>
# License: GPLv3

#import matplotlib
#matplotlib.use('WxAgg')
#matplotlib.interactive(True)
#import matplotlib.pyplot as plt

import numpy as np
import scipy.integrate as sci

from traits.api import HasTraits, Range, Instance, \
        on_trait_change, Button, Array
from traitsui.api import View, Item, HSplit, VSplit, Group

from tvtk.tools import visual
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
    SceneEditor

from pyface.timer.api import Timer
#from enable.component_editor import ComponentEditor

from chaco.api import ArrayPlotData, Plot
from chaco.chaco_plot_editor import ChacoPlotItem


def mass_spring_dashpot(y,t):
    # y[0] is the y-position
    # y[1] is the y-velocity
    # y[2] is the mass
    # y[3] is the damping const
    # y[4] is the spring constant
    # y[5] is the hammer force
    ny = np.zeros_like(y)
    ny[0] = y[1]
    ny[1] = -1.0 / y[2] * (y[3] * y[1] + y[4] * y[0] - y[5])
    ny[5] = 0.0
    return ny
    

################################################################################
# The visualization class
################################################################################
class MSDModel(HasTraits):

    # The parameters for the Lorenz system, defaults to the standard ones.
    m = Range(1.0, 20.0, 5.0, desc='m - m [kg]', enter_set=True, auto_set=False)
    c = Range(0.0, 250.0, 1.0, desc='c - c constant [Ns/m]', enter_set=True, auto_set=False)
    k = Range(10.0, 600.0, 100.0, desc='k - spring constant [N/m]', enter_set=True, auto_set=False)
    Fh = Range(0.0, 500.0, 50.0, desc = 'H - hammer force [N]', enter_set=True, auto_set = False)
#    Trec = Range(1.0, 10.0, 3.0, desc = 'number of seconds to record', enter_set = True, auto_set = False)
    sys_plot = Instance(Plot())
    chq_plot = Instance(Plot())
              
#    run = Button('Run system')
    hit = Button('Apply hammer')
    set_ics = Button('Set ICs')
    reset = Button('Reset')
    
    underdamped = Button('Underdamped system')
    critically_damped = Button('Critically damped system')
    overdamped = Button('Overdamped system')

    lambda1, lambda2 = 0.0, 0.0    
    
    timer = None
    dt = 50 # ms
    yt = np.array([0.0, 0.0, 5.0, 1.0, 1.0, 0.0, 0.0])
    apply_hammer = False
    
    # plotting data
    real_lam = Array()
    imag_lam = Array()
    tdata = Array()
    ydata = Array()
    
    rec_points = 600
    buf_pos = 0
    state_buf = np.zeros((rec_points, 3))

    # The mayavi(mlab) scene.
    scene = Instance(MlabSceneModel, args=())
    
    ########################################
    # The UI view to show the user.

    view = View(HSplit(
                    VSplit(
                        Item('scene',
                             editor = SceneEditor(scene_class=MayaviScene),
                             height = 400, width = 400,
                             show_label = False),
                        ChacoPlotItem('tdata', 'ydata',
                              title = 'Position & Velocity of system',
                              type = 'line',
                              x_auto = False, x_bounds = (0, 10),
                              y_auto = False, y_bounds = (-0.4, 0.4),
                              width = 600, height = 300)),
                    VSplit(
                        Group(
                            Item('m'),
                            Item('c'),
                            Item('k'),
                            Item('Fh'),
#                            Item('Trec'),
                            label = 'System parameters',
                            show_border = True),
                        Group(
                            Item('hit'),
                            Item('set_ics'),
                            Item('reset'),
                            label = 'Environment control',
                            show_border = True),
                        Group(
                            Item('underdamped'),
                            Item('critically_damped'),
                            Item('overdamped'),
                            label = 'Predefined parameters',
                            show_border = True),
                        Group(
                            ChacoPlotItem('real_lam', 'imag_lam',
                                          title = 'Characteristic equation roots',
                                          type = 'scatter',
                                          x_auto = False, x_bounds = (-50, 5),
                                          y_auto = False, y_bounds = (-30, 30),
                                          width = 400, height = 400))
                        )),
                    resizable = True,
                )

    ######################################################################
    # Trait handlers.
    ######################################################################

    @on_trait_change('scene.activated')
    def create_unicycle(self):
        visual.set_viewer(self.scene)
#        self.wheel = visual.cylinder(color = (0.2, 0.5, 0.5), pos = (-0.1, 0.4, 0), radius = 0.4, length = 0.2)
#        self.axle = visual.cylinder(color = (0.2, 0.5, 0.5), pos = (-0.2, 0.4, 0), radius = 0.05, length = 0.4)
#        self.supp1 = visual.box(color = (0.2, 0.5, 0.5), pos = (-0.18, 0.7, 0), length = 0.04, height = 0.6, width = 0.04)
#        self.supp2 = visual.box(color = (0.2, 0.5, 0.5), pos = (0.18, 0.7, 0), length = 0.04, height = 0.6, width = 0.04)
        self.wheel_top = visual.box(color = (0.2, 0.5, 0.5), pos = (0, 1.0, 0), length = 1.6, width = 1.6, height = 0.04)
        self.spring = visual.helix(coils = 8, axis = (0.0, 1.0, 0.0), color = (0.8, 0.2, 0.8), pos = (0.14, 1.0, 0), radius = 0.1, length = 0.5)
        self.car = visual.box(color = (0.2, 0.2, 0.8), pos = (0.0, 1.7, 0.0), length = 0.6, height = 0.4, width = 0.6)
        self.dash_top = visual.cylinder(axis = (0.0, -1.0, 0.0), color = (0.8, 0.8, 0.2), pos = (-0.14, 1.7, 0.0), radius = 0.1, length = 0.3)
        self.dash_bottom = visual.cylinder(axis = (0.0, 1.0, 0.0), color = (0.8, 0.8, 0.2), pos = (-0.14, 1.0, 0.0), radius = 0.05, length = 0.6)
        
#        self.rim1 = visual.cylinder(axis = (0.0, 0.0, 1.0), color = (0.0,0.0,0.0), pos = (0.0, 0.4, -0.35), radius = 0.11, length = 0.7)
#        self.rim2 = visual.cylinder(axis = (0.0, 1.0, 0.0), color = (0.0,0.0,0.0), pos = (0.0, 0.05, 0.0), radius = 0.11, length = 0.7)
#        self.road_a = visual.box(color = (0.2, 0.2, 0.2), pos = (0.0, -0.05, -0.5), length = 1.0, height = 0.1, width = 0.98)
#        self.road_b = visual.box(color = (0.2, 0.2, 0.2), pos = (0.0, -0.05, 0.5), length = 1.0, height = 0.1, width = 0.98)

        self.scene.camera.azimuth(45)
        self.scene.camera.elevation(20)
 
        self.recompute_char_eq()
        self.update_chq_data()
        
        self.tdata = self.state_buf[:,0]
        self.ydata = self.state_buf[:,1]

        self._run_fired()


    def move_road(self):
        """
        Simulate motion of the road and turning of the wheel.
        """
        w = self.road_a.width
        spd = 0.05
        if w > 2 * spd:
            self.road_a.z -= spd
            self.road_a.width -= spd * 2.0
            self.road_b.z -= spd
            self.road_b.width += spd * 2.0
        else:
            self.road_a.z = 0.0
            self.road_a.width = 1.98
            self.road_b.z = 1.0
            self.road_b.width = 0.0
            
        # also rotate my rims
        self.rim1.rotate(angle = 2 * np.pi * spd / 0.4 / (self.dt / 1000.0), axis = (1.0, 0.0, 0.0), origin = (0.0, 0.4, 0.0))
        self.rim2.rotate(angle = 2 * np.pi * spd / 0.4 / (self.dt / 1000.0), axis = (1.0, 0.0, 0.0), origin = (0.0, 0.4, 0.0))
            

    def integrate_system(self):
        """
        Run the system for one time step between frames.
        """
        yt = self.yt
        if self.apply_hammer:
            yt[5] = -self.Fh
            self.apply_hammer = False

            # store initial condition in buffer
            self.state_buf[:] = 0
            self.state_buf[0,0] = 0.0
            self.state_buf[0,1] = yt[0]
            self.state_buf[0,2] = yt[1]
            self.buf_pos = 1
        else:
            yt[5]= 0.0
            
        yt[2], yt[3], yt[4] = self.m, self.c, self.k
        yt1 = sci.odeint(mass_spring_dashpot, yt, [0.0, self.dt / 1000.0])[1,:]
        
        # move the m to reflect the integration
        if yt1[0] <= -0.45:
            self._reset_fired()
            yt1 = self.yt
            
        self.yt = yt1
        self.car.y = 1.7 + yt1[0]
        self.spring.length = 0.5 + yt1[0]
        self.dash_top.y = 1.7 + yt1[0]
        
        # store points and update
        bp = self.buf_pos
        if bp > 0 and bp < self.state_buf.shape[0]:
            self.state_buf[bp,0] = self.state_buf[bp-1, 0] + self.dt / 1000.0
            self.state_buf[bp,1] = yt1[0]
            self.state_buf[bp,2] = yt1[1]
            self.buf_pos += 1

            # update the chaco plot as well            
            self.tdata = self.state_buf[:bp,0]
            self.ydata = self.state_buf[:bp,1]
            self.trait_property_changed('ydata', self.ydata)
                
        elif bp == self.state_buf.shape[0]:
            self.buf_pos += 1
            self.tdata = self.state_buf[:,0]
            self.ydata = self.state_buf[:,1]
            self.trait_property_changed('ydata', self.ydata)
            
            
    def next_step(self):
        """
        Do all the necessary steps to proceed to the next animation frame.
        Turns off rendering during modifications to prevent CPU waste.
        """
        self.scene.disable_render = True        
#        self.move_road()
        self.integrate_system()
        self.scene.disable_render = False

    
    def _run_fired(self):
        """
        
        """
        if not self.timer:
            self.timer = Timer(self.dt, self.next_step)
            self.is_running = True
        else:
            if self.is_running:
                self.timer.Stop()
            else:
                self.timer.Start()
            self.is_running = False
            

    def _hit_fired(self):
        self.apply_hammer = True
        

    def _reset_fired(self):
        self.yt = np.array([0.0, 0.0, self.m, self.c, self.k, 0.0, 0.0])
        self.buf_pos = 0
        

    def _m_changed(self):
        self.recompute_char_eq()
        self.update_chq_data()
        

    def _c_changed(self):
        self.recompute_char_eq()
        self.update_chq_data()
        

    def _k_changed(self):
        self.recompute_char_eq()
        self.update_chq_data()
        

#    def _Trec_fired(self):
#        self.rec_points = int(self.Trec * 1000.0 / self.dt)
#        self.state_buf = np.zeros((self.rec_points, 3))
        

    def recompute_char_eq(self):
        # read system params
        m = self.m
        c = self.c
        k = self.k
        
        # reparametrize
        w0_sq = k / m
        p = c / (2*m)
        
        # analyze system
        self.omega0 = np.sqrt(w0_sq)
        if p**2 < w0_sq:
            self.omega1 = np.sqrt(w0_sq - p**2)
        else:
            self.omega1 = 0.0
            
        self.lambda1 = -p - np.sqrt(complex(p**2 - w0_sq))
        self.lambda2 = -p + np.sqrt(complex(p**2 - w0_sq))


    def _overdamped_fired(self):
        self.m  = 5
        self.c  = 130
        self.k  = 100
        self.Fh = 400
    

    def _critically_damped_fired(self):
        self.m = 4
        self.c = 40
        self.k = 100
        self.Fh = 180
#        self.Trec = 3.0
    

    def _underdamped_fired(self):
        self.m = 5
        self.c = 4.0
        self.k = 100
        self.Fh= 70
#        self.Trec = 6.0
        

    def _set_ics_fired(self):
        self.yt[0] = 0.2
        self.yt[1]= 0.0

        # store initial condition in buffer
        self.buf_pos = 1
        self.ydata[:] = 0.0
        self.state_buf[0,0] = 0.0
        self.state_buf[0,1] = self.yt[0]
        self.state_buf[0,2] = 0.0        
                

    def update_chq_data(self):
        l1, l2 = self.lambda1, self.lambda2
        self.real_lam = np.array([np.real(l1), np.real(l2)])
        self.imag_lam = np.array([np.imag(l1), np.imag(l2)])

        
if __name__ == '__main__':
    # Instantiate the class and configure its traits.
    msdm = MSDModel()
    msdm.configure_traits()

