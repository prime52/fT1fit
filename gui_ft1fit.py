# coding=utf-8
'''
Read pre and post dicom files.
Display ECV map.

contributors: Yoon-Chul Kim, Khu Rai Kim
'''

import sys
import PyQt5.QtGui as qt
import PyQt5.QtCore as qc
import PyQt5.QtWidgets as qw

from PyQt5 import sip

import pickle, time, os
import pydicom as dicom
import copy
import numpy as np
import SimpleITK as sitk

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import calc_t1map_cpp as t1
import calc_t1map_py as t1_py


class Window(qw.QDialog):

    def plot(self, figure, canvas, image, clim, colorbar_, title, etime):

        # print('plot() called.. '+title)
        ax = figure.add_subplot(111)
        ax.clear()
        colorbar_.remove()
        cax = ax.imshow(image, cmap='jet', clim=(0, clim))
        ax.set_title(title + ", elapsed time = " + str(round(etime, 2)) + " (sec)")

        figure.colorbar(cax, ticks=range(0, clim, round(clim/10)))

        canvas.draw()

        return

    def plot_pre(self, t1map, title, etime):
        # ax = self.pre_fig.add_subplot(111)
        # ax.clear()

        self.pre_fig_ax.clear()
        self.pre_fig_cb.remove()

        clim = self.pre_clim
        cax = self.pre_fig_ax.imshow(t1map, cmap='jet', clim=(0, clim))
        self.pre_fig_ax.set_title(title + ", elapsed time = " + str(round(etime, 2)) + " (sec)")
        self.pre_fig_cb = self.pre_fig.colorbar(cax, ticks=range(0, clim, round(clim/10)))

        self.pre_canvas.draw()

        return

    def plot_post(self, t1map, title, etime):
        # ax = self.post_fig.add_subplot(111)
        # ax.clear()

        self.post_fig_ax.clear()
        self.post_fig_cb.remove()

        clim = self.post_clim
        cax = self.post_fig_ax.imshow(t1map, cmap='jet', clim=(0, clim))
        self.post_fig_ax.set_title(title + ", elapsed time = " + str(round(etime, 2)) + " (sec)")
        self.post_fig_cb = self.post_fig.colorbar(cax, ticks=range(0, clim, round(clim/10)))

        self.post_canvas.draw()

        return

    def plot_ecvmap(self, ecvmap):
        # ax = self.ecv_fig.add_subplot(111)
        # ax.clear()

        self.ecv_fig_ax.clear()
        self.ecv_fig_cb.remove()

        clim = 100
        cax = self.ecv_fig_ax.imshow(ecvmap, cmap='jet', clim=(0, clim))
        self.ecv_fig_ax.set_title("ECV map")

        self.ecv_fig_cb = self.ecv_fig.colorbar(cax, ticks=range(0, 100, 10))

        self.ecv_canvas.draw()

        return

    def construct_figure(self, click_event_handler):
        figure = Figure()
        ax = figure.add_subplot(1,1,1)
        canvas = FigureCanvas(figure)
        cax = ax.imshow(np.zeros((384, 288)), cmap='gray')
        colorbar_ = figure.colorbar(cax)
        canvas.mpl_connect("button_press_event", click_event_handler)

        return figure, canvas, ax, colorbar_


    def construct_figure_plot(self, click_event_handler):
        figure = Figure(figsize=(2,4))
        ax = figure.add_subplot(1,1,1)
        canvas = FigureCanvas(figure)
        # ax.plot(np.zeros((384, 288)), cmap='gray')
        canvas.mpl_connect("button_press_event", click_event_handler)

        return figure, canvas, ax


    def click_event_handler(self, event, title):

        if event.ydata is None or event.xdata is None:
            return

        event_x = int(np.around(event.ydata))
        event_y = int(np.around(event.xdata))

        if title == 'pre':
            t1_val = self.t1map_pre[event_x, event_y]
            # print(a)
            # ax = self.pre_fig_ax
            self.pre_fig_ax.set_title(title + " T1 = " + str(int(t1_val)) + " (ms) at (x,y)=(" + str(event_y) + ", " + str(event_x) + ")")
            p0 = self.pre_fig_ax.scatter(event_y, event_x, c="black")
            self.pre_canvas.draw()
            p0.remove()

            # plot T1 curve.
            scatter_x = self.invtime
            scatter_y = self.image[event_x, event_y, :len(self.invtime)]

            f = np.vectorize(t1.func_orig)

            plot_begin = self.invtime[0]
            plot_end = self.invtime[-1]
            plot_x = np.linspace(plot_begin, plot_end, 100)

            a = self.t1_params_pre[event_x, event_y, 0]
            b = self.t1_params_pre[event_x, event_y, 1]
            c = self.t1_params_pre[event_x, event_y, 2]

            plot_y = np.abs(f(plot_x, a, b, c))
            # t1_val = (1 / b) * (a / (a + c) - 1)

            ax = self.t1_fig_ax
            ax.clear()
            ax.plot(plot_x, plot_y, c="royalblue", linewidth=2.0)
            ax.scatter(scatter_x, scatter_y, c="orange")
            ax.set_xlabel("Inversion time (ms)")
            ax.set_ylabel("Magnitude")
            ax.set_title("T1: " + str(int(t1_val)) + " (ms)")
            # #ax.set_ylim([None, self.t1_max])
            ax.grid()

            self.t1_canvas.draw()

        elif title == 'post':
            t1_val = self.t1map_post[event_x, event_y]
            self.post_fig_ax.set_title(title + " T1 = " + str(round(t1_val)) + " (ms) at (x,y)=(" + str(event_y) + ", " + str(event_x) + ")")
            # self.post_fig_ax.plot(event_y, event_x, 'ro')
            p0 = self.post_fig_ax.scatter(event_y, event_x, c="black")
            self.post_canvas.draw()
            p0.remove()

            # plot T1 curve.
            scatter_x = self.invtime_post
            scatter_y = self.image_post[event_x, event_y, :len(self.invtime_post)]

            f = np.vectorize(t1.func_orig)

            plot_begin = self.invtime_post[0]
            plot_end = self.invtime_post[-1]
            plot_x = np.linspace(plot_begin, plot_end, 100)

            a = self.t1_params_post[event_x, event_y, 0]
            b = self.t1_params_post[event_x, event_y, 1]
            c = self.t1_params_post[event_x, event_y, 2]

            plot_y = np.abs(f(plot_x, a, b, c))

            ax = self.t1_fig_ax
            ax.clear()
            ax.plot(plot_x, plot_y, c="royalblue", linewidth=2.0)
            ax.scatter(scatter_x, scatter_y, c="orange")
            ax.set_xlabel("Inversion time (ms)")
            ax.set_ylabel("Magnitude")
            ax.set_title("T1: " + str(int(t1_val)) + " (ms)")
            ax.grid()

            self.t1_canvas.draw()

        elif title == 'ecv':
            a = self.ecvmap[event_x, event_y]
            self.ecv_fig_ax.set_title(" ECV = " + str(round(a)) + " (%) at (x,y)=(" + str(event_y) + ", " + str(event_x) + ")")
            p0 = self.ecv_fig_ax.scatter(event_y, event_x, c="black")
            self.ecv_canvas.draw()
            p0.remove()

        return


    def open_file_action_dicom(self):

        self.series = []

        # direc_default = os.getcwd()
        # direc_default = r'D:/data/cardiacMR'

        self.data_directory = r'C:\Users\yoonc\Documents\journal_writing\t1map_cpp\test_t1gui_exe\data\subj9\pre'
        print(self.data_directory)
        self.dirname = str(qw.QFileDialog.getExistingDirectory(self, "Select Directory", self.data_directory))
        print(self.dirname)
        self.data_directory = self.dirname
        session_files = os.listdir(self.dirname)
        print(session_files)
        for i, f in enumerate(session_files):
            self.series.append(f)
        nfiles = i+1
        print(nfiles)
        for j, f in enumerate(session_files):
            print(j, f)
            ds = dicom.read_file(self.dirname+'/'+f)
            if j == 0:
                # print(ds.Rows, ds.Columns)
                pixelDims = (int(ds.Rows), int(ds.Columns), int(nfiles))
                # print(pixelDims)
                self.ir_img = np.zeros(pixelDims, dtype=ds.pixel_array.dtype)
                self.invtime = np.zeros(int(nfiles), dtype=ds.pixel_array.dtype)

            self.ir_img[:, :, ds.InstanceNumber-1] = ds.pixel_array
            self.invtime[ds.InstanceNumber-1] = ds.InversionTime

        self.image = self.ir_img

        self.invtime = list(filter(lambda x: x != 0, sorted(self.invtime)))

        # print(ds)
        print(self.invtime)

        txt1 = '\n Manufacturer: ' + ds.Manufacturer + \
               '\n Magnetic Field Strength: ' + str(ds.MagneticFieldStrength) +\
               '\n Manufacturer Model Name: ' + ds.ManufacturerModelName +\
               '\n Series Description: ' + ds.SeriesDescription +\
                '\n Study Date: ' + ds.StudyDate +\
               '\n Acquisition Time: ' + ds.AcquisitionTime +\
               '\n Image Dimensions: (' + str(ds.Rows) + ', ' + str(ds.Columns) + ')' + \
            '\n Inversion Time (ms): ' + str(self.invtime) +\
            '\n Pixel Spacing: (' + str(ds.PixelSpacing[0]) + ', ' + str(ds.PixelSpacing[1]) + ')'
            # '\n Acquisition Date: ' + ds.AcquisitionDate +\



        self.pre_header.setText(txt1)

        return


    def open_file_action_dicom_post(self):

        self.series_post = []

        # direc_default = os.getcwd()
        # direc_default = r'D:/data/cardiacMR'

        # print(self.dirname)
        self.dirname = str(qw.QFileDialog.getExistingDirectory(self, "Select Directory", self.data_directory))
        print(self.dirname)
        self.data_directory = self.dirname
        session_files = os.listdir(self.dirname)
        print(session_files)
        for i, f in enumerate(session_files):
            self.series.append(f)
        nfiles = i+1
        print(nfiles)
        for j, f in enumerate(session_files):
            print(j, f)
            ds = dicom.read_file(self.dirname+'/'+f)

            if j == 0:
                # print(ds.Rows, ds.Columns)
                pixelDims = (int(ds.Rows), int(ds.Columns), int(nfiles))
                # print(pixelDims)
                self.ir_img_post = np.zeros(pixelDims, dtype=ds.pixel_array.dtype)
                self.invtime_post = np.zeros(int(nfiles), dtype=ds.pixel_array.dtype)

            self.ir_img_post[:, :, ds.InstanceNumber-1] = ds.pixel_array
            self.invtime_post[ds.InstanceNumber-1] = ds.InversionTime

        self.image_post = self.ir_img_post
        self.invtime_post = list(filter(lambda x: x != 0, sorted(self.invtime_post)))
        # print(self.invtime_post)


        txt1 = '\n Manufacturer: ' + ds.Manufacturer + \
               '\n Magnetic Field Strength: ' + str(ds.MagneticFieldStrength) +\
               '\n Manufacturer Model Name: ' + ds.ManufacturerModelName +\
               '\n Series Description: ' + ds.SeriesDescription +\
                '\n Study Date: ' + ds.StudyDate +\
               '\n Acquisition Time: ' + ds.AcquisitionTime +\
               '\n Image Dimensions: (' + str(ds.Rows) + ', ' + str(ds.Columns) + ')' + \
            '\n Inversion Time (ms): ' + str(self.invtime_post) +\
            '\n Pixel Spacing: (' + str(ds.PixelSpacing[0]) + ', ' + str(ds.PixelSpacing[1]) + ')'
            # '\n Acquisition Date: ' + ds.AcquisitionDate +\


        self.post_header.setText(txt1)

        return

    def plot_t1_map(self, t1_params, title, etime):
        a = t1_params[:, :, 0]
        b = t1_params[:, :, 1]
        c = t1_params[:, :, 2]
        t1 = (1 / b) * (a / (a + c) - 1)

        if title == "pre":
            print('pre plot t1')
            # self.t1_cpp = t1
            # self.plot(self.pre_fig, self.pre_canvas, t1, self.pre_clim, self.pre_fig_cb, title, etime)

            self.plot_pre(t1, title, etime)

            self.t1map_pre = t1

        elif title == "post":
            # self.t1_py = t1
            # self.plot(self.post_fig, self.post_canvas, t1, self.post_clim, self.post_fig_cb, title, etime)
            self.plot_post(t1, title, etime)
            self.t1map_post = t1

        # elif title == "ecv":
        #     print('pyrd plot t1')
        #     self.t1_pyrd = t1
        #     self.plot(self.ecv_fig, self.ecv_canvas, t1, self.pyrd_clim, title, etime)
        else:
            print("title name is not in the list")

        return

    def set_clim_pre(self):
        print(self.etime_pre)
        # self.plot(self.pre_fig, self.pre_canvas, self.t1map_pre, self.pre_clim, "pre", self.etime_pre)
        self.plot_pre(self.t1map_pre, "pre", self.etime_pre)
        return

    def set_clim_post(self):
        print(self.etime_post)
        # self.plot(self.post_fig, self.post_canvas, self.t1map_post, self.post_clim, "post", self.etime_post)
        self.plot_post(self.t1map_post, "post", self.etime_post)
        return

    def btn1_calc_t1map(self):
        # pre
        self.calc_method_str = self.combo1.currentText()

        if self.calc_method_str == "C++ multi-core RD-NLS":
            # print('C++, pre T1')
            stime = time.time()
            self.t1_params_pre = t1.calculate_T1map_cpp_rd(self.ir_img, self.invtime, multicore_flag=1)
            self.etime_pre = time.time() - stime
            self.plot_t1_map(self.t1_params_pre, title="pre", etime=self.etime_pre)

        elif self.calc_method_str == "C++ single-core RD-NLS":
            # print('C++, pre T1')
            stime = time.time()
            self.t1_params_pre = t1.calculate_T1map_cpp_rd(self.ir_img, self.invtime, multicore_flag=0)
            self.etime_pre = time.time() - stime
            self.plot_t1_map(self.t1_params_pre, title="pre", etime=self.etime_pre)

        elif self.calc_method_str == "C++ multi-core LM":
            # print('C++, pre T1')
            stime = time.time()
            self.t1_params_pre = t1.calculate_T1map_cpp_lm(self.ir_img, self.invtime, multicore_flag=1)
            self.etime_pre = time.time() - stime
            self.plot_t1_map(self.t1_params_pre, title="pre", etime=self.etime_pre)

        elif self.calc_method_str == "C++ single-core LM":
            # print('C++, pre T1')
            stime = time.time()
            self.t1_params_pre = t1.calculate_T1map_cpp_lm(self.ir_img, self.invtime, multicore_flag=0)
            self.etime_pre = time.time() - stime
            self.plot_t1_map(self.t1_params_pre, title="pre", etime=self.etime_pre)

        elif self.calc_method_str == "python LM":
            # print('pure python running')
            # print(self.ir_img.shape)
            # print(self.invtime)
            stime = time.time()
            self.t1_params_pre = t1_py.calculate_T1map(self.ir_img, self.invtime)
            self.etime_pre = time.time() - stime
            self.plot_t1_map(self.t1_params_pre, title="pre", etime=self.etime_pre)

        elif self.calc_method_str == "python RD-NLS":
            stime = time.time()
            invtime = np.asarray(self.invtime)
            self.t1_params_pre = t1_py.calculate_T1map_pyrd(self.ir_img, invtime)
            # print('pyrd done()')
            # print(self.pyrd_t1_params.shape)
            self.etime_pre = time.time() - stime
            self.plot_t1_map(self.t1_params_pre, title="pre", etime=self.etime_pre)

        return

    def btn2_calc_t1map(self):
        # post

        if self.calc_method_str == "C++ multi-core RD-NLS":
            # print('C++, pre T1')
            stime = time.time()
            self.t1_params_post = t1.calculate_T1map_cpp_rd(self.ir_img_post, self.invtime_post, multicore_flag=1)
            self.etime_post = time.time() - stime
            self.plot_t1_map(self.t1_params_post, title="post", etime=self.etime_post)

        elif self.calc_method_str == "C++ single-core RD-NLS":
            # print('C++, pre T1')
            stime = time.time()
            self.t1_params_post = t1.calculate_T1map_cpp_rd(self.ir_img_post, self.invtime_post, multicore_flag=0)
            self.etime_post = time.time() - stime
            self.plot_t1_map(self.t1_params_post, title="post", etime=self.etime_post)

        if self.calc_method_str == "C++ multi-core LM":
            stime = time.time()
            self.t1_params_post = t1.calculate_T1map_cpp_lm(self.ir_img_post, self.invtime_post, multicore_flag=1)
            self.etime_post = time.time() - stime
            self.plot_t1_map(self.t1_params_post, title="post", etime=self.etime_post)

        elif self.calc_method_str == "C++ single-core LM":
            stime = time.time()
            self.t1_params_post = t1.calculate_T1map_cpp_lm(self.ir_img_post, self.invtime_post, multicore_flag=0)
            self.etime_post = time.time() - stime
            self.plot_t1_map(self.t1_params_post, title="post", etime=self.etime_post)

        elif self.calc_method_str == "python LM":
            print('python running')
            stime = time.time()
            self.t1_params_post = t1_py.calculate_T1map(self.ir_img_post, self.invtime_post)
            self.etime_post = time.time() - stime
            self.plot_t1_map(self.t1_params_post, title="post", etime=self.etime_post)

        elif self.calc_method_str == "python RD-NLS":
            stime = time.time()
            invtime = np.asarray(self.invtime_post)
            self.t1_params_post = t1_py.calculate_T1map_pyrd(self.ir_img_post, invtime)
            # print('pyrd done()')
            # print(self.pyrd_t1_params.shape)
            self.etime_post = time.time() - stime
            self.plot_t1_map(self.t1_params_post, title="post", etime=self.etime_post)

        return

    def imregister_sitk_nonrigid(self, fixed, moving):

        fixed_image = sitk.Cast(sitk.GetImageFromArray(fixed), sitk.sitkFloat32)
        moving_image = sitk.Cast(sitk.GetImageFromArray(moving), sitk.sitkFloat32)

        transformDomainMeshSize = [2] * moving_image.GetDimension()

        tx = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=10, maximumNumberOfCorrections=10,
                               maximumNumberOfFunctionEvaluations=100, costFunctionConvergenceFactor=1e+7)
        R.SetInitialTransform(tx, True)
        R.SetInterpolator(sitk.sitkLinear)
        # R.AddCommand(sitkIterationEvent, lambda: command_iteration(R))
        outTx = R.Execute(fixed_image, moving_image)

        print("imregister_sitk_nonrigid() done...")

        return outTx, fixed_image, moving_image

    def btn3_calc_ecvmap(self):
        # ecv
        print('ecv map calculation')

        self.ecv_method_str = self.combo2.currentText()
        t1map_pre = self.t1map_pre

        print(self.ecv_method_str)

        if self.ecv_method_str == "No Registration":
            print('no register')
            t1map_post = self.t1map_post

        elif self.ecv_method_str == "Registration":
            print('sitk registration')
            max_ind_pre = len(self.invtime)-2
            max_ind_post = len(self.invtime_post)-1
            irlast_pre = self.ir_img[:, :, max_ind_pre]
            irlast_post = self.ir_img_post[:, :, max_ind_post]
            nrow, ncol = irlast_post.shape
            invtime_post = np.asarray(self.invtime_post)
            irimg_post_moved = np.zeros((nrow, ncol, len(np.nonzero(invtime_post)[0])))
            Tx1, fixed, moving = self.imregister_sitk_nonrigid(irlast_pre, irlast_post)
            rs1 = sitk.ResampleImageFilter()
            rs1.SetReferenceImage(fixed)
            rs1.SetInterpolator(sitk.sitkLinear)
            rs1.SetTransform(Tx1)
            for tino in np.nonzero(invtime_post)[0]:
                moving1 = sitk.Cast(sitk.GetImageFromArray(self.ir_img_post[:, :, tino]), sitk.sitkFloat32)
                irimg_post_moved[:, :, tino] = sitk.GetArrayFromImage(sitk.Cast(sitk.RescaleIntensity(rs1.Execute(moving1)), sitk.sitkUInt16))

            ''' calculate post T1 map again. '''
            if self.combo1.currentText() == "C++ multi-core RD-NLS":
                t1_params = t1.calculate_T1map_cpp_rd(irimg_post_moved, invtime_post, multicore_flag=1)
            if self.combo1.currentText() == "C++ single-core RD-NLS":
                t1_params = t1.calculate_T1map_cpp_rd(irimg_post_moved, invtime_post, multicore_flag=0)
            elif self.combo1.currentText() == "C++ multi-core LM":
                t1_params = t1.calculate_T1map_cpp_lm(irimg_post_moved, invtime_post, multicore_flag=1)
            elif self.combo1.currentText() == "C++ single-core LM":
                t1_params = t1.calculate_T1map_cpp_lm(irimg_post_moved, invtime_post, multicore_flag=0)
            elif self.calc_method_str == "python LM":
                t1_params = t1_py.calculate_T1map(irimg_post_moved, invtime_post)
            elif self.calc_method_str == "python RD-NLS":
                t1_params = t1_py.calculate_T1map_pyrd(irimg_post_moved, invtime_post)

            a = t1_params[:, :, 0]
            b = t1_params[:, :, 1]
            c = t1_params[:, :, 2]
            t1map_post = (1 / b) * (a / (a + c) - 1)

        else:
            print(" invalid method selection ")

        ''' blood segmentation '''
        img_for_seg = copy.copy(t1map_pre)
        hematocrit = 0.42
        method_lvblood = 1
        if method_lvblood == 1:
            ''' method 1, threshold '''
            thresh_t1 = 1200
            mask_blood = np.zeros(img_for_seg.shape)
            mask_blood[img_for_seg > thresh_t1] = 1
            # mask_blood_erode = sm.erosion(mask_blood, sm.disk(5))
            ind_blood = np.where(mask_blood == 1)
        elif method_lvblood == 2:
            ''' method 2, seeded region growing using Simple ITK '''
            seed = (int(self.lvc[0]), int(self.lvc[1]))
            img1 = sitk.GetImageFromArray(img_for_seg, isVector=True)
            seg_rg = sitk.ConnectedThreshold(img1, seedList=[seed], lower=1000, upper=1600)
            mask_rg = sitk.GetArrayFromImage(seg_rg)
            mask_blood = np.zeros(img_for_seg.shape)
            mask_blood[mask_rg > 0] = 1
            ind_blood = np.where(mask_blood == 1)

        print(img_for_seg.shape)
        print(t1map_post.shape)

        sig_blood_pre = img_for_seg[ind_blood]
        R1_pre = 1 / t1map_pre
        sig_blood_post = t1map_post[ind_blood]
        R1_post = 1 / t1map_post

        T1_blood_pre = np.median(sig_blood_pre.flatten())
        T1_blood_post = np.median(sig_blood_post.flatten())
        if T1_blood_post > 1000:
            T1_blood_post = 350
        elif T1_blood_post == 0:
            T1_blood_post = 350

        print('T1 blood pre = %4.1f, T1 blood post = %4.1f' % (T1_blood_pre, T1_blood_post))

        R1_blood_post = 1 / T1_blood_post
        R1_blood_pre = 1 / T1_blood_pre

        ecvmap = 100 * (1 - hematocrit) * (R1_post - R1_pre) / (R1_blood_post - R1_blood_pre)
        ecvmap = np.nan_to_num(ecvmap)
        ecvmap[np.where(ecvmap > 100)] = 100.0
        ecvmap[np.where(ecvmap < 0)] = 0.0

        self.ecvmap = ecvmap

        self.plot_ecvmap(ecvmap)

        return

    def combo1_methodchange(self):
        self.calc_method_str = self.combo1.currentText()
        print(self.calc_method_str)

    def combo2_methodchange(self):
        self.ecv_method_str = self.combo2.currentText()

    def combo_pre_clim_methodchange(self):
        str1 = self.combo_pre_clim.currentText()

        if str1 == "500 ms":
            self.pre_clim = 500
        elif str1 == "1000 ms":
            self.pre_clim = 1000
        elif str1 == "1500 ms":
            self.pre_clim = 1500
        elif str1 == "2000 ms":
            self.pre_clim = 2000
        else:
            print('limit not in the list.')

        self.set_clim_pre()

        return

    def combo_post_clim_methodchange(self):
        str1 = self.combo_post_clim.currentText()

        if str1 == "500 ms":
            self.post_clim = 500
        elif str1 == "600 ms":
            self.post_clim = 600
        elif str1 == "800 ms":
            self.post_clim = 800
        elif str1 == "1000 ms":
            self.post_clim = 1000
        else:
            print('limit not in the list.')

        self.set_clim_post()

        return

    def __init__(self, parent=None):

        super(Window, self).__init__(parent)
        self.t1_fig, self.t1_canvas, self.t1_fig_ax = self.construct_figure_plot(lambda event: None)
        self.pre_fig,  self.pre_canvas, self.pre_fig_ax, self.pre_fig_cb = self.construct_figure(lambda event: self.click_event_handler(event, "pre"))
        self.post_fig, self.post_canvas, self.post_fig_ax, self.post_fig_cb = self.construct_figure(lambda event: self.click_event_handler(event, "post"))
        self.ecv_fig,  self.ecv_canvas, self.ecv_fig_ax, self.ecv_fig_cb = self.construct_figure(lambda event: self.click_event_handler(event, "ecv"))

        layout = qw.QGridLayout()

        self.data_directory = os.getcwd()

        self.menubar = qw.QMenuBar(self)
        self.menubar.setStyleSheet("QMenuBar {font-size: 11pt; font-family: Arial; background-color:rgb(200, 200, 200); color: blue;}")

        self.open_file = self.menubar.addMenu("Open DICOM: pre T1")
        open_action = qw.QAction("Open File", self)
        open_action.triggered.connect(self.open_file_action_dicom)
        self.open_file.addAction(open_action)

        self.open_file2 = self.menubar.addMenu("Open DICOM: post T1")
        open_action2 = qw.QAction("Open File", self)
        open_action2.triggered.connect(self.open_file_action_dicom_post)
        self.open_file2.addAction(open_action2)

        self.combo1 = qw.QComboBox()
        self.combo1.setStyleSheet(
            "QComboBox {font-size: 11pt; font-family: Arial; background-color:rgb(128, 128, 128); color: white;}")

        self.combo1.addItems(
            ["C++ multi-core RD-NLS", "C++ multi-core LM", "C++ single-core RD-NLS", "C++ single-core LM", "python RD-NLS",
             "python LM"])
        self.combo1.currentIndexChanged.connect(self.combo1_methodchange)

        self.combo2 = qw.QComboBox()
        self.combo2.setStyleSheet(
            "QComboBox {font-size: 11pt; font-family: Arial; background-color:rgb(128, 128, 128); color: white;}")
        self.combo2.addItems(["No Registration", "Registration"])
        self.combo2.currentIndexChanged.connect(self.combo2_methodchange)

        self.btn1 = qw.QPushButton("Calculate Pre T1 Map!")
        self.btn1.setStyleSheet("QPushButton {font-size: 14pt; font-family: Arial; background-color:rgb(255, 128, 0); color: white;}")
        self.btn1.clicked.connect(self.btn1_calc_t1map)

        self.btn2 = qw.QPushButton("Calculate Post T1 Map!")
        self.btn2.setStyleSheet("QPushButton {font-size: 14pt; font-family: Arial; background-color:rgb(128, 0, 128); color: white;}")
        self.btn2.clicked.connect(self.btn2_calc_t1map)

        self.btn3 = qw.QPushButton("Calculate ECV Map!")
        self.btn3.setStyleSheet("QPushButton {font-size: 14pt; font-family: Arial; background-color:rgb(0, 128, 255); color: white;}")
        self.btn3.clicked.connect(self.btn3_calc_ecvmap)

        self.pre_clim = 1500
        self.post_clim = 1000

        self.combo_pre_clim = qw.QComboBox()
        self.combo_pre_clim.addItems(["500 ms", "1000 ms", "1500 ms", "2000 ms"])
        self.combo_pre_clim.currentIndexChanged.connect(self.combo_pre_clim_methodchange)

        self.combo_post_clim = qw.QComboBox()
        self.combo_post_clim.addItems(["500 ms", "600 ms", "800 ms", "1000 ms"])
        self.combo_post_clim.currentIndexChanged.connect(self.combo_post_clim_methodchange)

        self.pre_header = qw.QTextEdit()
        self.post_header = qw.QTextEdit()

        layout.addWidget(self.menubar, 0, 0)
        layout.addWidget(self.combo1, 0, 1)
        layout.addWidget(self.combo2, 0, 2)
        layout.addWidget(self.btn1, 1, 0)
        layout.addWidget(self.btn2, 1, 1)
        layout.addWidget(self.btn3, 1, 2)

        layout.addWidget(self.pre_canvas, 2, 0)
        layout.addWidget(self.post_canvas, 2, 1)
        layout.addWidget(self.ecv_canvas, 2, 2)

        layout.addWidget(self.combo_pre_clim, 3, 0)
        layout.addWidget(self.combo_post_clim, 3, 1)

        layout.addWidget(self.pre_header, 4, 0)
        layout.addWidget(self.post_header, 4, 1)

        layout.addWidget(self.t1_canvas, 4, 2)

        self.setLayout(layout)


if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())
