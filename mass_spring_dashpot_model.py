# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:07:22 2012

@author: martin
"""

"""
This code creates, simulates and animates a free vibrations system (spring,
dashpot, mass) in a car unicycle model.
"""
# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2008-2009, Enthought, Inc.
# License: BSD Style.

import matplotlib
matplotlib.use('WxAgg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt

import numpy as np
import scipy.integrate as sci

from traits.api import HasTraits, Range, Instance, \
        on_trait_change, Button, Complex, Float
from traitsui.api import View, Item, HSplit, VSplit, Group

from tvtk.tools import visual
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
    SceneEditor

from pyface.timer.api import Timer


def mass_spring_dashpot(y,t):
    # y[0] is the y-position
    # y[1] is the y-velocity
    # y[2] is the mass
    # y[3] is the damping const
    # y[4] is the spring constant
    # y[5] is the hammer force
    # y[6] is the controller gain
    ny = np.zeros_like(y)
    ny[0] = y[1]
    ny[1] = -1.0 / y[2] * (y[3] * y[1] + y[4] * y[0] - y[6] * y[0] - y[5])
    ny[5] = 0.0
    return ny
    

################################################################################
# The visualization class
################################################################################
class UnicycleModel(HasTraits):

    # The parameters for the Lorenz system, defaults to the standard ones.
    mass = Range(1.0, 20.0, 5.0, desc='m - mass [kg]', enter_set=True,
              auto_set=False)
    damping = Range(0.0, 250.0, 1.0, desc='c - damping constant [Ns/m]', enter_set=True,
              auto_set=False)
    spring_constant = Range(10.0, 2000.0, 100.0, desc='k - spring constant [N/m]', enter_set=True,
              auto_set=False)
    controller_gain = Range(-500.0, 500.0, 0.0, desc='K - controller gain', enter_set=True,
                            auto_set=False)
    hammer_force = Range(0.0, 500.0, 50.0, desc = 'H - hammer force [N]', enter_set=True,
              auto_set = False)
    recording_time = Range(1.0, 10.0, 1.0, desc = 'number of seconds to record', enter_set = True,
                        auto_set = False)
              
#    run = Button('Run system')
    hit = Button('Apply hammer')
    set_ics = Button('Set ICs')
    reset = Button('Reset')
    
    underdamped = Button('Underdamped system')
    critically_damped = Button('Critically damped system')
    overdamped = Button('Overdamped system')
    
    lambda1 = Complex(desc = 'first root of the char. eq.')
    lambda2 = Complex(desc = 'second root of the char. eq.')
    omega0 = Float(desc = 'frequency of undamped system')
    omega1 = Float(desc = 'frequency of damped system')
    
    timer = None
    dt = 50 # ms
    yt = np.array([0.0, 0.0, 5.0, 1.0, 1.0, 0.0, 0.0])
    apply_hammer = False
    
    # state buffer (100 points)
    state_buf = np.zeros((60, 3))
    buf_pos = 0

    # The mayavi(mlab) scene.
    scene = Instance(MlabSceneModel, args=())
    
    ########################################
    # The UI view to show the user.
    view = View(HSplit(
                    Group(
                        Item('scene',
                             editor = SceneEditor(scene_class=MayaviScene),
                             height = 600, width = 600,
                             show_label = False)),
                    VSplit(
                    Group(
                        Item('mass'),
                        Item('damping'),
                        Item('spring_constant'),
 #                       Item('controller_gain'),
                        Item('hammer_force'),
                        Item('recording_time'),
                        label = 'System parameters',
                        show_border = True),
                    Group(
#                        Item('run'),
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
                        Item('lambda1', style = 'readonly'),
                        Item('lambda2', style = 'readonly'),
                        Item('omega0', style = 'readonly'),
                        Item('omega1', style = 'readonly'),
                        label = 'Char. eq. roots',
                        show_border = True))
                    ),
                    resizable = True
                )

    ######################################################################
    # Trait handlers.
    ######################################################################

    @on_trait_change('scene.activated')
    def create_unicycle(self):
        visual.set_viewer(self.scene)
        self.wheel = visual.cylinder(color = (0.2, 0.5, 0.5), pos = (-0.1, 0.4, 0), radius = 0.4, length = 0.2)
        self.axle = visual.cylinder(color = (0.2, 0.5, 0.5), pos = (-0.2, 0.4, 0), radius = 0.05, length = 0.4)
        self.supp1 = visual.box(color = (0.2, 0.5, 0.5), pos = (-0.18, 0.7, 0), length = 0.04, height = 0.6, width = 0.04)
        self.supp2 = visual.box(color = (0.2, 0.5, 0.5), pos = (0.18, 0.7, 0), length = 0.04, height = 0.6, width = 0.04)
        self.wheel_top = visual.box(color = (0.2, 0.5, 0.5), pos = (0, 1.0, 0), length = 0.8, width = 0.8, height = 0.04)
        self.spring = visual.helix(coils = 8, axis = (0.0, 1.0, 0.0), color = (0.8, 0.2, 0.8), pos = (0.14, 1.0, 0), radius = 0.1, length = 0.5)
        self.car = visual.box(color = (0.2, 0.2, 0.8), pos = (0.0, 1.7, 0.0), length = 0.6, height = 0.4, width = 0.6)
        self.dash_top = visual.cylinder(axis = (0.0, -1.0, 0.0), color = (0.8, 0.8, 0.2), pos = (-0.14, 1.7, 0.0), radius = 0.1, length = 0.3)
        self.dash_bottom = visual.cylinder(axis = (0.0, 1.0, 0.0), color = (0.8, 0.8, 0.2), pos = (-0.14, 1.0, 0.0), radius = 0.05, length = 0.6)
        
        self.rim1 = visual.cylinder(axis = (0.0, 0.0, 1.0), color = (0.0,0.0,0.0), pos = (0.0, 0.4, -0.35), radius = 0.11, length = 0.7)
        self.rim2 = visual.cylinder(axis = (0.0, 1.0, 0.0), color = (0.0,0.0,0.0), pos = (0.0, 0.05, 0.0), radius = 0.11, length = 0.7)
        
        self.road_a = visual.box(color = (0.2, 0.2, 0.2), pos = (0.0, -0.05, -0.5), length = 1.0, height = 0.1, width = 0.98)
        self.road_b = visual.box(color = (0.2, 0.2, 0.2), pos = (0.0, -0.05, 0.5), length = 1.0, height = 0.1, width = 0.98)
        
#        self.pos_arrow = visual.box(color = (0.8, 0.8, 0.5), pos = (-1.0, 1.25, 0.2), length = 0.04, width = 0.04, height = 0.5)
#        self.vel_arrow = visual.box(color = (0.2, 0.8, 0.5), pos = (-1.0, 1.25, 0.4), length = 0.04, width = 0.04, height = 0.0)
        
        self.scene.camera.azimuth(45)
        self.scene.camera.elevation(20)
        self.recompute_char_eq()
        self._run_fired()

    def move_road(self):
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
        yt = self.yt
        if self.apply_hammer:
            yt[5] = -self.hammer_force
            self.apply_hammer = False

            # store initial condition in buffer
            self.buf_pos = 1
            sb = self.state_buf
            sb[0,0] = 0.0
            sb[0,1] = yt[0]
            sb[0,2] = yt[1]
        else:
            yt[5]= 0.0
            
        yt[2], yt[3], yt[4], yt[6] = self.mass, self.damping, self.spring_constant, self.controller_gain
        yt1 = sci.odeint(mass_spring_dashpot, yt, [0.0, self.dt / 1000.0])[1,:]
        
        # move the mass to reflect the integration
        if yt1[0] <= -0.45:
            self._reset_fired()
            yt1 = self.yt
            
        self.yt = yt1
        self.car.y = 1.7 + yt1[0]
        self.spring.length = 0.5 + yt1[0]
        self.dash_top.y = 1.7 + yt1[0]
        
#        self.pos_arrow.y = 1.0 + (0.5 + yt1[0]) / 2.0
#        self.pos_arrow.height = 0.5 + yt1[0]
#        
#        self.vel_arrow.y = 1.0 + (0.5 + yt1[1]) / 2.0
#        self.vel_arrow.height = np.abs(yt1[1])

        # store points and update
        bp = self.buf_pos
        sb = self.state_buf
        if bp > 0 and bp < sb.shape[0]:
            sb[bp,0] = sb[bp-1,0] + self.dt / 1000.0
            sb[bp,1] = yt1[0]
            sb[bp,2] = yt1[1]
            self.buf_pos += 1
        elif bp == sb.shape[0]:
            self.update_mpl_plots()
            self.buf_pos = 0
            
            
    def next_step(self):
        self.scene.disable_render = True        
        self.move_road()
        self.integrate_system()
        self.scene.disable_render = False

    
    def _run_fired(self):
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
        self.yt = np.array([0.0, 0.0, self.mass, self.damping, self.spring_constant, 0.0, 0.0])
        
    def _mass_changed(self):
        self.recompute_char_eq()
        
    def _damping_changed(self):
        self.recompute_char_eq()
        
    def _spring_constant_changed(self):
        self.recompute_char_eq()
        
    def _controller_gain_changed(self):
        self.recompute_char_eq()
        
    def recompute_char_eq(self):
        # read system params
        m = self.mass
        c = self.damping
        k = self.spring_constant
        K = self.controller_gain
        
        # reparametrize
        w0_sq = (k - K) / m
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
        self.mass = 5
        self.damping = 130
        self.spring_constant = 100
        self.hammer_force= 400
        self.recording_time = 3.0
    
    def _critically_damped_fired(self):
        self.mass = 5
        self.damping = 45
        self.spring_constant = 100
        self.hammer_force= 180
        self.recording_time = 3.0
    
    def _underdamped_fired(self):
        self.mass = 5
        self.damping = 4.0
        self.spring_constant = 100
        self.hammer_force= 70
        self.recording_time = 6.0
        
    def _set_ics_fired(self):
        self.yt[0] = 0.2
        self.yt[1]= 0.0

        # store initial condition in buffer
        sb = self.state_buf
        self.buf_pos = 1
        sb[0,0] = 0.0
        sb[0,1] = self.yt[0]
        sb[0,2] = self.yt[1]
        
    def update_mpl_plots(self):
        plt.figure(1)
        plt.clf()
        
        sb = self.state_buf
        plt.subplot(211)
        plt.title('Position & Velocity of mass')
        plt.plot(sb[:,0], sb[:,1], 'b-', linewidth = 1.5)
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.subplot(212)
        plt.plot(sb[:,0], sb[:,2], 'g-', linewidth = 1.5)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        

    def _recording_time_changed(self):
        self.state_buf = np.zeros((int(self.recording_time * 20), 3))
        

if __name__ == '__main__':
    # Instantiate the class and configure its traits.
    um = UnicycleModel()
    um.configure_traits()

