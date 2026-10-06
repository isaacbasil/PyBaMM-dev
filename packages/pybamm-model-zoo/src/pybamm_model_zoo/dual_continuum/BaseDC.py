import pybamm
import numpy as np

class BaseDC:
    '''A base dual-continuum model. 
    '''

    def __init__(self, 
                 name,
                 cell_type, 
                 var_pts, 
                 param_values, 
                 order, 
                 prev_correction_terms, 
                 calc_soc, 
                 calc_c_surf_a_priori, 
                 transient_inputs,  
                 sei, 
                 min_eta_plating, 
                 plating_correction_closure):

        self.order = order
        self.var_pts = var_pts
        self.calc_soc = calc_soc
        self.cell_type = cell_type
        self.param_values = param_values
        self.calc_c_surf_a_priori = calc_c_surf_a_priori
        self.prev_correction_term_n, self.prev_correction_term_p = prev_correction_terms
        self.transient_inputs = transient_inputs
        self.sei = sei
        self.min_eta_plating = min_eta_plating
        self.plating_correction_closure = plating_correction_closure


        # although 0 order does not need a priori calc, set to true to avoid solving for c_surf
        if self.order == 0:
            self.calc_c_surf_a_priori = True
        # a priori concentration calc not available for Yang
        if self.order == "Yang":
            self.calc_c_surf_a_priori = False  
           

        # create a base model
        self.model = pybamm.BaseModel()

        # number of particle radii
        self.radii_count = None

        # initialise spatial vars
        self.x_n = None
        self.x_s = None
        self.x_p = None
        self.r_p = None
        self.mesh = None

        # intialise vars
        self.c_e_s = None
        self.phi_e_s = None
        self.c_s_p = None
        self.phi_s_p = None
        self.phi_s_p_surf = None
        self.eta_p = None
        self.c_e_p = None
        self.c_e = None
        self.phi_e = None
        self.phi_e_p = None

        # initialise params
        self.W = None
        self.H = None
        self.A_cs = None
        self.i_app = None
        self.Ah = None
        self.u0_p = None
        self.i0_p = None
        self.vf_am_p = None
        self.av_p_real = None
        self.j_p = None
        self.i0_lifoil = None

        # params unique to DC model
        self.c_s_p_surf = None
        self.correction_term_p = None
        self.j_p_ave = None
        self.u0_p_bar = None
        self.s0_p_surface_average = None

            
        self.T = None
        self.F = None
        self.R = None
        self.sigma_e_eff = None
        self.theta = None
        self.sigma_s_eff_p = None
        self.transp_eff_e = None
        self.transp_eff_e_p = None
        self.D_e_eff = None
        self.D_s_p = None
        self.eps_e = None
        self.eps_e_p = None
        self.c_p_max = None
        self.k_p = None
        self.sep_surf_por = None
        self.transp_no = None
        self.u0_p_init = None
        self.L_n = None
        self.L_s = None
        self.L_p = None
        self.V_p = None
        self.I = None
        self.u0_p_bulk = None

        if self.cell_type == "Full cell":
            self.c_s_n = None
            self.phi_s_n = None
            self.eta_n = None
            self.c_e_n = None
            self.phi_e_n = None
            self.u0_n = None
            self.i0_n = None
            self.vf_am_n = None
            self.av_n_real = None
            self.j_n = None
            self.c_s_n_surf = None
            self.j_n_bar = None
            self.j_n_ave = None
            self.j0_n_bar = None 
            self.eta_n_bar = None
            self.u0_n_bar = None
            self.s0_n_surface_average = None
            self.sigma_s_eff_n = None
            self.transp_eff_e_n = None
            self.D_s_n = None
            self.eps_e_n = None
            self.c_n_max = None
            self.k_n = None
            self.u0_n_init = None
            self.u0_n_bulk = None



        if self.sei: 
            self.j_sei = None
            self.D_sei = None
            self.c_sei_ref = None
            self.sigma_sei = None 
            self.omega_sei = None
            self.V_mol_sei = None
            self.coeff_stoich_sei = None 
            self.R_sei = None
            self.L_sei_0 = None

    def define_variables(self):
        '''Define the variables for the model'''

        # spatial variables
        self.x_n = pybamm.SpatialVariable("x_n", domain=["negative electrode"], coord_sys="cartesian")
        self.x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
        self.x_p = pybamm.SpatialVariable("x_p", domain=["positive electrode"], coord_sys="cartesian")

        # define separator varaiables
        self.c_e_s = pybamm.Variable("Separator electrolyte concentration [mol.m-3]", domain="separator")
        self.phi_e_s = pybamm.Variable("Separator electrolyte potential [V]", domain="separator")

        # define positive electrode variables

        self.phi_s_p = pybamm.Variable("Positive electrode potential [V]", domain="positive electrode")

        self.c_e_p = pybamm.Variable("Positive electrolyte concentration [mol.m-3]", domain="positive electrode")
        self.phi_e_p = pybamm.Variable("Positive electrolyte potential [V]", domain="positive electrode")


        self.c_s_p = pybamm.Variable("R-averaged positive particle concentration [mol.m-3]", domain="positive electrode") # to match the DFN key
        if not self.calc_c_surf_a_priori:
            self.c_s_p_surf = pybamm.Variable("Positive particle surface concentration [mol.m-3]", domain="positive electrode")


        if self.cell_type == "Full cell":
            self.phi_s_n = pybamm.Variable("Negative electrode potential [V]", domain="negative electrode")
            self.c_e_n = pybamm.Variable("Negative electrolyte concentration [mol.m-3]", domain="negative electrode")
            self.phi_e_n = pybamm.Variable("Negative electrolyte potential [V]", domain="negative electrode")
            self.c_e = pybamm.concatenation(self.c_e_n, self.c_e_s, self.c_e_p)
            self.phi_e = pybamm.concatenation(self.phi_e_n, self.phi_e_s, self.phi_e_p)
            self.c_s_n = pybamm.Variable("R-averaged negative particle concentration [mol.m-3]", domain="negative electrode") # to match the DFN key
            if not self.calc_c_surf_a_priori:
                self.c_s_n_surf = pybamm.Variable("Negative particle surface concentration [mol.m-3]", domain="negative electrode")
        else:
            self.c_e = pybamm.concatenation(self.c_e_s, self.c_e_p)
            self.phi_e = pybamm.concatenation(self.phi_e_s, self.phi_e_p)

        self.t = pybamm.t

        # ===== SEI ======= 
        if self.sei:
            if self.cell_type == "Full cell":
                self.L_sei = pybamm.Variable("SEI thickness [m]", domain="negative electrode") 
                # need to add j as variable because it is now implicitly defined (depends on itself)
                self.j_n = pybamm.Variable((f"Negative electrode interfacial current density [A.m-2]"), domain=f"negative electrode")
            else:
                self.L_sei = pybamm.Variable("SEI thickness [m]", domain="positive electrode")
                # need to add j as variable because it is now implicitly defined (depends on itself)
                self.j_p = pybamm.Variable((f"Positive electrode interfacial current density [A.m-2]"), domain=f"positive electrode")
            

    def define_parameters(self):
        '''A function to set all parameters from the import parameter file'''

        # adding physical params to dictionairy
        self.R = pybamm.Parameter('Ideal gas constant [J.K-1.mol-1]')
        self.F = pybamm.Parameter('Faraday constant [C.mol-1]')

        # simulation params
        self.H = pybamm.Parameter('Electrode height [m]')
        self.W = pybamm.Parameter('Electrode width [m]')
        self.A_cs = self.W * self.H # cross sectional area
        self.i_app = pybamm.Parameter("Current density [A.m-2]")
        self.I = self.i_app * self.A_cs

        # global params
        self.T = pybamm.Parameter('Temperature [K]')
        self.F_RT = self.F / (self.R * self.T)

        # effective params used?
        try:
            eff_props = bool(self.param_values["Effective properties"])
        except KeyError:
            eff_props = False

        # define separator params
        eps_sep = pybamm.Parameter('Separator porosity')
        self.sep_surf_por = pybamm.Parameter('Separator surface porosity')
        tau_sep = pybamm.Parameter("Separator tortuosity (electrolyte)")

        # define electrolyte parameters
        self.c_e_0 = pybamm.Parameter('Initial concentration in electrolyte [mol.m-3]')
        try:
            float(self.param_values['Thermodynamic factor'])  # will throw error if D is function
            tdf = pybamm.Parameter('Thermodynamic factor')
        except TypeError:
            tdf_inputs = {"Electrolyte concentration [mol.m-3]": self.c_e, "Temperature [K]": self.T}
            tdf = pybamm.FunctionParameter("Thermodynamic factor", tdf_inputs)
        try:
            float(self.param_values['Electrolyte diffusivity [m2.s-1]'])  # will throw error if D is function
            D_e = pybamm.Parameter('Electrolyte diffusivity [m2.s-1]')
        except TypeError:
            D_e_inputs = {"Electrolyte concentration [mol.m-3]": self.c_e, "Temperature [K]": self.T}
            D_e = pybamm.FunctionParameter("Electrolyte diffusivity [m2.s-1]", D_e_inputs)
        try:
            float(self.param_values['Electrolyte conductivity [S.m-1]'])
            sigma_e = pybamm.Parameter('Electrolyte conductivity [S.m-1]')
        except TypeError:
            sigma_e_inputs = {"Electrolyte concentration [mol.m-3]": self.c_e, "Temperature [K]": self.T}
            sigma_e = pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", sigma_e_inputs)

        self.transp_no = pybamm.Parameter('Cation transference number')
        self.theta = - 2 * self.R * self.T * (1 - self.transp_no) / self.F * tdf

        # define positive electrode params
        eps_s_p = pybamm.Parameter('Positive electrode active material volume fraction')
        eps_e_p = pybamm.Parameter('Positive electrode porosity')
        self.c_p_max = pybamm.Parameter('Maximum concentration in positive electrode [mol.m-3]')
        self.vf_am_p = pybamm.Parameter("Positive electrode active material volume fraction")
        self.c_s_p_0 = pybamm.Parameter('Initial concentration in positive electrode [mol.m-3]')

        try:
            float(self.param_values['Positive electrode diffusivity [m2.s-1]'])
            self.D_s_p = pybamm.Parameter('Positive electrode diffusivity [m2.s-1]')
        except TypeError:
            D_s_p_inputs = {"Positive electrode concentration [mol.m-3]": self.c_s_p,
                            "Temperature [K]": self.T,
                            }
            self.D_s_p = pybamm.FunctionParameter('Positive electrode diffusivity [m2.s-1]', D_s_p_inputs)

        if self.cell_type == "Full cell":
            self.L_n = pybamm.Parameter('Negative electrode thickness [m]')
            eps_s_n = pybamm.Parameter('Negative electrode active material volume fraction')
            eps_e_n = pybamm.Parameter('Negative electrode porosity')
            self.c_n_max = pybamm.Parameter('Maximum concentration in negative electrode [mol.m-3]')
            self.vf_am_n = pybamm.Parameter("Negative electrode active material volume fraction")
            self.c_s_n_0 = pybamm.Parameter('Initial concentration in negative electrode [mol.m-3]')
            self.V_n = self.A_cs * self.L_n
            try:
                float(self.param_values['Negative electrode diffusivity [m2.s-1]'])
                self.D_s_n = pybamm.Parameter('Negative electrode diffusivity [m2.s-1]')
            except TypeError:
                D_s_n_inputs = {"Negative electrode concentration [mol.m-3]": self.c_s_n,
                                "Temperature [K]": self.T,
                                }
                self.D_s_n = pybamm.FunctionParameter('Negative electrode diffusivity [m2.s-1]', D_s_n_inputs)
        else:
            self.L_n = pybamm.Scalar(0) # define Li foil properties

        # define geometry params
        self.L_s = pybamm.Parameter('Separator thickness [m]')
        self.L_p = pybamm.Parameter('Positive electrode thickness [m]')
        self.L_x = self.L_n + self.L_s + self.L_p
        self.V_p = self.A_cs * self.L_p

        if eff_props:
            self.sigma_s_eff_p = pybamm.Parameter("Positive electrode effective conductivity (electrode)")
            transp_eff_e_p = pybamm.Parameter("Positive electrode electrolyte transport efficiency")
            if self.cell_type == "Full cell":
                self.sigma_s_eff_n = pybamm.Parameter("Negative electrode effective conductivity (electrode)")
                transp_eff_e_n = pybamm.Parameter("Negative electrode electrolyte transport efficiency")
        else:
            sigma_p = pybamm.Parameter('Positive electrode conductivity [S.m-1]')
            tau_s_p = pybamm.Parameter('Positive electrode tortuosity (electrode)')
            tau_e_p = pybamm.Parameter('Positive electrode tortuosity (electrolyte)')
            self.sigma_s_eff_p = eps_s_p / tau_s_p * sigma_p
            transp_eff_e_p = eps_e_p / tau_e_p
            if self.cell_type == "Full cell":
                sigma_n = pybamm.Parameter('Negative electrode conductivity [S.m-1]')
                tau_s_n = pybamm.Parameter('Negative electrode tortuosity (electrode)')
                tau_e_n = pybamm.Parameter('Negative electrode tortuosity (electrolyte)')
                self.sigma_s_eff_n = eps_s_n / tau_s_n * sigma_n
                transp_eff_e_n = eps_e_n / tau_e_n


        self.prob_frac_area = pybamm.Scalar(1)
        self.prob_frac_vol = pybamm.Scalar(1)

        # effective properties
        transp_eff_e_sep = pybamm.PrimaryBroadcast((eps_sep / tau_sep), "separator")
        eps_sep = pybamm.PrimaryBroadcast(eps_sep, "separator")

        self.eps_e_p = pybamm.PrimaryBroadcast(eps_e_p, "positive electrode")
        self.transp_eff_e_p = pybamm.PrimaryBroadcast(transp_eff_e_p, "positive electrode")
        self.sigma_s_eff_p = pybamm.PrimaryBroadcast(self.sigma_s_eff_p, "positive electrode")
        
        if self.cell_type == "Full cell":
            self.eps_e_n = pybamm.PrimaryBroadcast(eps_e_n, "negative electrode")
            self.transp_eff_e_n = pybamm.PrimaryBroadcast(transp_eff_e_n, "negative electrode")
            self.sigma_s_eff_n = pybamm.PrimaryBroadcast(self.sigma_s_eff_n, "negative electrode")
            self.transp_eff_e = pybamm.concatenation(self.transp_eff_e_n, transp_eff_e_sep, self.transp_eff_e_p)
            self.eps_e = pybamm.concatenation(self.eps_e_n, eps_sep, self.eps_e_p)
        else:
            self.transp_eff_e = pybamm.concatenation(transp_eff_e_sep, self.transp_eff_e_p)
            self.eps_e = pybamm.concatenation(eps_sep, self.eps_e_p)

        self.D_e_eff = self.transp_eff_e * D_e
        self.sigma_e_eff = self.transp_eff_e * sigma_e


        # real specific surface area calculations
        self.av_p_real = pybamm.Parameter("Positive electrode specific surface area from image (AM-electrolyte) [m-1]")
        if "Positive electrode specific surface area from image (AM-CBD) [m-1]" in self.param_values.keys():
            cbd_surf_por = pybamm.Parameter("CBD surface porosity")
            a_p_am_cbd = pybamm.Parameter("Positive electrode specific surface area from image (AM-CBD) [m-1]")
            # multiply by surface porosity
            self.av_p_real += cbd_surf_por * a_p_am_cbd
        if "Positive electrode specific surface area from image (AM-separator) [m-1]" in self.param_values.keys():
            a_p_am_sep = pybamm.Parameter("Positive electrode specific surface area from image (AM-separator) [m-1]")
            # multiply by surface porosity
            self.av_p_real += self.sep_surf_por * a_p_am_sep
        if self.cell_type == "Full cell":
            try:
                self.av_n_real = pybamm.Parameter("Negative electrode specific surface area from image (AM-electrolyte) [m-1]") 
            except KeyError:
                pass 
            if "Negative electrode specific surface area from image (AM-CBD) [m-1]" in self.param_values.keys():
                a_n_am_cbd = pybamm.Parameter("Negative electrode specific surface area from image (AM-CBD) [m-1]")
                self.av_n_real += cbd_surf_por * a_n_am_cbd # multiply by surface porosity

        # ======== SEI ========
        if self.sei: 
            self.L_sei_0 = pybamm.Parameter("SEI initial thickness [m]")
            self.D_sei = pybamm.Parameter("SEI diffusion coefficient [m2.s-1]")
            self.c_sei_ref = pybamm.Parameter("SEI reference concentration [mol.m-3]")
            self.sigma_sei = pybamm.Parameter("SEI conductivity [S.m-1]")
            self.omega_sei = pybamm.Parameter("SEI fitting parameter")
            self.V_mol_sei = pybamm.Parameter("SEI molar volume [m3.mol-1]")
            self.coeff_stoich_sei = pybamm.Parameter("SEI stoichiometric coefficient")
            self.R_sei = self.L_sei / self.sigma_sei
        # ======================

        # ====== Li Plating =====
        # TODO: update to cover full cell
        if self.min_eta_plating and self.plating_correction_closure: 
            #self.min_p_surf = pybamm.Parameter("Positive electrode minimum p surface") 
            self.b_min = pybamm.Parameter("Positive electrode minimum b separator surface") 

        # =======================


        self.k_p = pybamm.Parameter("Positive electrode rate constant")
        u0_p_init_inputs = {"Positive electrode SOC": (self.c_s_p_0 / self.c_p_max)}
        self.u0_p_init = pybamm.FunctionParameter("Positive electrode OCP [V]", u0_p_init_inputs)
        
        if self.cell_type == "Full cell":
            self.k_n = pybamm.Parameter("Negative electrode rate constant")
            u0_n_init_inputs = {"Negative electrode SOC": (self.c_s_n_0 / self.c_n_max)}
            self.u0_n_init = pybamm.FunctionParameter("Negative electrode OCP [V]", u0_n_init_inputs)
        else:
            i0_lifoil_inputs = {"Electrolyte concentration [mol.m-3]": pybamm.boundary_value(self.c_e, "left"),
                            "Lithium concentration [mol.m-3]": 1,
                            "Temperature [K]": self.T, # meaningless, just required for Pybamm consistency
                            }
            self.i0_lifoil = pybamm.FunctionParameter("Exchange-current density for lithium metal electrode [A.m-2]",
                                                    i0_lifoil_inputs)


        # -------defining 2eq surface concentration-------
        self.j_p_ave = pybamm.PrimaryBroadcast(self.i_app / (self.L_p * self.av_p_real), "positive electrode")

        if self.transient_inputs:
            self.s0_p_surface_average = pybamm.FunctionParameter("Positive electrode s0 surface average transient", {"Time [s]": self.t})
        elif "Positive electrode s0 surface average dimensionless" in self.param_values:
            self.s0_p_surface_average = pybamm.Parameter("Positive electrode s0 surface average dimensionless") * self.L_p / (self.D_s_p * self.F)
        else:
            self.s0_p_surface_average =  pybamm.Parameter("Positive electrode s0 surface average") 
        

        if self.order == 0:
            self.c_s_p_surf = self.c_s_p
            self.correction_term_p = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "positive electrode")
        elif self.order == 1:
            if self.calc_c_surf_a_priori:

                if self.transient_inputs:
                    self.correction_term_p = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "positive electrode")
                    s = {}
                    s1 = {}
                    for key, value in self.transient_inputs.items():
                        t_s = value["t_s"]
                        t_input = self.t - t_s
                        s[key] = pybamm.PrimaryBroadcast(pybamm.FunctionParameter("Positive electrode s0 surface average transient", {"Time [s]": t_input}), "positive electrode")
                        # first subtract decay term from previous epoch
                        j_p_ave_previous = pybamm.PrimaryBroadcast(value["j_p_ave_previous"], "positive electrode")
                        self.correction_term_p += - j_p_ave_previous * s[key]
                        # now add contribution from the current density of the current epoch 
                        epoch_j_ave = pybamm.PrimaryBroadcast(value["epoch i_app"] / (self.L_p * self.av_p_real), "positive electrode")
                        self.correction_term_p += epoch_j_ave * s[key]

                    self.c_s_p_surf = self.c_s_p + self.correction_term_p

                else:
                    self.correction_term_p = self.s0_p_surface_average * self.j_p_ave
                    self.c_s_p_surf = self.c_s_p + self.correction_term_p
            else:
                pass # implicit function for c_surf and j will be solved

        elif self.order == "Yang":    

            if "Positive particle radius [m]" in self.param_values: # incase MP model defined
                self.l_s_p = pybamm.Parameter("Positive particle radius [m]")
            elif "Positive particle mean radius [m]" in self.param_values:
                self.l_s_p = pybamm.Parameter("Positive particle mean radius [m]")
            else:
                raise ValueError("Positive particle radius parameter not found for Yang's model")
            a_d_p = pybamm.Parameter("Positive electrode Yang fitting parameter")
            small_perturbation = 1e-200
            self.t_dif_p = self.l_s_p ** 2 / self.D_s_p
            self.t_cut_p = self.t_dif_p / (a_d_p ** 2)
            regime_p = (2 * pybamm.t < 2 * self.t_cut_p) # multiply by 2 to avoid a bug with the treatment of the heaviside
            self.l_dif_p = (a_d_p * (self.D_s_p * (self.t + small_perturbation))**0.5) * regime_p + self.l_s_p * (1 - regime_p)
        
        else:
            raise ValueError("Order input is not supported")
        
        if self.cell_type == "Full cell":

            self.j_n_ave = pybamm.PrimaryBroadcast(- self.i_app / (self.L_n * self.av_n_real), "negative electrode")

            if self.transient_inputs:
                self.s0_n_surface_average = pybamm.FunctionParameter("Negative electrode s0 surface average transient", {"Time [s]": self.t})
            elif "Negative electrode s0 surface average dimensionless" in self.param_values:
                self.s0_n_surface_average = pybamm.Parameter("Negative electrode s0 surface average dimensionless") * self.L_n / (self.D_s_n * self.F)
            else:
                self.s0_n_surface_average =  pybamm.Parameter("Negative electrode s0 surface average") 

            if self.order == 0:
                self.c_s_n_surf = self.c_s_n
                self.correction_term_n = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "negative electrode")
            elif self.order == 1:
                if self.calc_c_surf_a_priori:
                    # Transient section NOT validated for full cell
                    if self.transient_inputs:
                        self.correction_term_n = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "negative electrode")
                        s = {}
                        for key, value in self.transient_inputs.items():
                            t_s = value["t_s"]
                            t_input = self.t - t_s
                            s[key] = pybamm.PrimaryBroadcast(pybamm.FunctionParameter("Negative electrode s0 surface average transient", {"Time [s]": t_input}), "negative electrode")
                            # first subtract decay term from previous epoch
                            j_n_ave_previous = pybamm.PrimaryBroadcast(value["j_n_ave_previous"], "negative electrode")
                            self.correction_term_n += - j_n_ave_previous * s[key]
                            # now add contribution from the current density of the current epoch 
                            epoch_j_ave = pybamm.PrimaryBroadcast(- value["epoch i_app"] / (self.L_n * self.av_n_real), "negative electrode")
                            self.correction_term_n += epoch_j_ave * s[key]
                        self.c_s_n_surf = self.c_s_n + self.correction_term_n

                    else:
                        self.correction_term_n = self.s0_n_surface_average * self.j_n_ave
                        self.c_s_n_surf = self.c_s_n + self.correction_term_n
                else:
                    pass # implicit function for c_surf and j will be solved


        # -------OCP and i0 functions---------------


    def set_model_equations(self):
        '''Sets governing equations, not including ICs and BCs'''

        # ==== SEI ====
        if self.sei:
            g0 = 1 / 25 # libat choice
            if self.cell_type == "Full cell": 
                g_sei = (1 - self.F_RT * self.omega_sei * self.R_sei * self.j_n)  
                chi_sei = (pybamm.sin(np.pi / 2 * pybamm.Minimum(pybamm.Maximum(0, g_sei/ g0), 1))) ** 2               
                self.j_sei = - self.F * self.D_sei * self.c_sei_ref / self.L_sei * pybamm.exp(- self.F_RT * (self.phi_s_n - self.phi_e_n - self.R_sei * self.j_n)) * g_sei * chi_sei
                self.eta_n = self.phi_s_n - self.phi_e_n - self.u0_n - self.R_sei * self.j_n
                self.j_int = self.j_bv(self.i0_n, self.eta_n, self.T)
                # add j to alegraic equations if sei model is used  
                self.model.algebraic[self.j_n] = self.j_n - (self.j_int + self.j_sei)
            else:
                g_sei = (1 - self.F_RT * self.omega_sei * self.R_sei * self.j_p)
                chi_sei = (pybamm.sin(np.pi / 2 * pybamm.Minimum(pybamm.Maximum(0, g_sei/ g0), 1))) ** 2
                self.j_sei = - self.F * self.D_sei * self.c_sei_ref / self.L_sei * pybamm.exp(- self.F_RT * (self.phi_s_p - self.phi_e_p - self.R_sei * self.j_p)) * g_sei * chi_sei
                self.eta_p = self.phi_s_p - self.phi_e_p - self.u0_p - self.R_sei * self.j_p
                self.j_int = self.j_bv(self.i0_p, self.eta_p, self.T)
                # add j to alegraic equations if sei model is used
                self.model.algebraic[self.j_p] = self.j_p - (self.j_int + self.j_sei)

            # add thickness ODE 
            dL_sei_dt = - self.V_mol_sei / (self.F * self.coeff_stoich_sei) * self.j_sei
            self.model.rhs[self.L_sei] = dL_sei_dt
        # ============

        # ======= define reaction rates and source terms =====

        j_s = pybamm.PrimaryBroadcast(0, "separator")             

        if not self.sei or self.cell_type == "Full cell": # for half cell, eta_p and j_p defined above when sei included
            self.eta_p = self.phi_s_p - self.phi_e_p - self.u0_p 
            self.j_p = self.j_bv(self.i0_p, self.eta_p, self.T)

        a_j_p = self.av_p_real * self.j_p


        if self.cell_type == "Full cell":
            if not self.sei:
                self.eta_n = self.phi_s_n - self.phi_e_n - self.u0_n
                self.j_n = self.j_bv(self.i0_n, self.eta_n, self.T)
            a_j_n = self.av_n_real * self.j_n
            a_j = pybamm.concatenation(a_j_n, j_s, a_j_p)
        else:
            a_j = pybamm.concatenation(j_s, a_j_p)


        i_e = - self.sigma_e_eff * (pybamm.grad(self.phi_e) + self.theta / self.c_e * pybamm.grad(self.c_e))

        i_p = - self.sigma_s_eff_p * pybamm.grad(self.phi_s_p)  # positive electrode (macroscale) current

        N_e = - self.D_e_eff * pybamm.grad(self.c_e)  # not actually true electrolyte flux

        # define electrolyte mass balance
        dc_e_dt = 1 / self.eps_e * (-pybamm.div(N_e) + (
                1 - self.transp_no) * a_j / self.F)  # define the rhs equation, assuming averaged conc represents surface conc.
        self.model.rhs[self.c_e] = dc_e_dt  # add the equation to rhs dictionary

        # define positive electrode mass balance
        if self.sei and self.cell_type != "Full cell":
            dc_s_p_dt = - 1/(self.vf_am_p * self.F) * self.av_p_real * self.j_int # different j when sei modelled
        else:
            dc_s_p_dt = - 1/(self.vf_am_p * self.F) * a_j_p 

        self.model.rhs[self.c_s_p] = dc_s_p_dt

        # define electrolyte charge balance
        self.model.algebraic[self.phi_e] = self.L_x ** 2 * (- pybamm.div(i_e) + a_j)

        # define positive electrode charge balance
        self.model.algebraic[self.phi_s_p] = self.L_x ** 2 * (- pybamm.div(i_p) - a_j_p)

        # define algebraic eq for c_surf if needed
        if not self.calc_c_surf_a_priori:
            if self.order == 1: 
                # smooth at switch between epochs to ease stiffness
                if self.prev_correction_term_p is None:
                    self.correction_term_p = self.j_p * self.s0_p_surface_average
                else: 
                    self.correction_term_p = self.smooth_correction_term(self.j_p, self.s0_p_surface_average, self.prev_correction_term_p)

            elif self.order == "Yang":
                self.correction_term_p = - (self.l_dif_p / (2 * self.l_s_p) - self.l_dif_p**2 / (6 * self.l_s_p**2)) * (self.j_p * self.l_s_p / (self.D_s_p * self.F)) # from paper

            # ============= Key line for standard DC model ============
            self.model.algebraic[self.c_s_p_surf] = self.c_s_p_surf - (self.c_s_p + self.correction_term_p)
            # ========================================================


        if self.cell_type == "Full cell":
            if self.sei:
                dc_s_n_dt = - 1/(self.vf_am_n * self.F) * self.av_n_real * self.j_int # different j when sei modelled
            else:
                dc_s_n_dt = - 1/(self.vf_am_n * self.F) * a_j_n 
            self.model.rhs[self.c_s_n] = dc_s_n_dt
            i_n = - self.sigma_s_eff_n * pybamm.grad(self.phi_s_n) 
            self.model.algebraic[self.phi_s_n] = self.L_x ** 2 * (- pybamm.div(i_n) - a_j_n)

            if not self.calc_c_surf_a_priori:
                if self.order == 1:
                    if self.prev_correction_term_n is None:
                        self.correction_term_n = self.j_n * self.s0_n_surface_average
                    else: 
                        self.correction_term_n = self.smooth_correction_term(self.j_n, self.s0_n_surface_average, self.prev_correction_term_n)

                    # ============= Key line for standard DC model ============
                    self.model.algebraic[self.c_s_n_surf] = self.c_s_n_surf - (self.c_s_n + self.correction_term_n)
                    # ========================================================

    def set_initial_and_boundary_conditions(self):

        # Initial conditions must also be provided for algebraic equations, but they are not really initial conditions,
        # but rather an initial guess for a root-finding algorithm which calculates consistent initial conditions
        if self.cell_type == "Full cell":
            self.model.initial_conditions[self.c_s_n] = self.c_s_n_0
            self.model.initial_conditions[self.phi_s_n] = pybamm.Scalar(0)
            self.model.initial_conditions[self.phi_s_p] = self.u0_p_init 
            self.model.initial_conditions[self.phi_e] = - self.u0_n_init
        else:
            self.model.initial_conditions[self.phi_s_p] = self.u0_p_init
            self.model.initial_conditions[self.phi_e] =  pybamm.Scalar(0)
        self.model.initial_conditions[self.c_e] = self.c_e_0     

        # ======= SEI =========
        if self.sei:
            self.model.initial_conditions[self.L_sei] = self.L_sei_0
            if self.cell_type == "Full cell":
                # initial guess for j
                self.model.initial_conditions[self.j_n] = self.j_n_ave
            else:
                self.model.initial_conditions[self.j_p] = self.j_p_ave

        # ======= define boundary conditions ============
        if self.cell_type == "Full cell":
            self.model.boundary_conditions[self.c_s_n] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
            self.model.boundary_conditions[self.phi_s_n] = {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
            self.model.boundary_conditions[self.c_e] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
            self.model.boundary_conditions[self.phi_e] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        else:
            grad_c_lifoil = pybamm.boundary_value(((1 / self.D_e_eff) * (1 - self.transp_no) * self.i_app / self.F),
                                                  "left")

            phi_e_lbc = - 2 * self.R * self.T / self.F * pybamm.arcsinh(- self.i_app / (self.sep_surf_por * 2 * self.i0_lifoil))

            self.model.boundary_conditions[self.c_e] = {
                "left": (grad_c_lifoil, "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }

            self.model.boundary_conditions[self.phi_e] = {
                "left": (phi_e_lbc, "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }

        

        self.model.initial_conditions[self.c_s_p] = self.c_s_p_0


        grad_phi_rhs = self.i_app / pybamm.boundary_value(self.sigma_s_eff_p, "right")
        self.model.boundary_conditions[self.phi_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (grad_phi_rhs, "Neumann"),
        }

        if not self.calc_c_surf_a_priori:
            self.model.initial_conditions[self.c_s_p_surf] = self.c_s_p_0
            self.model.boundary_conditions[self.c_s_p_surf] = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),}
            
            if self.cell_type == "Full cell":
                self.model.initial_conditions[self.c_s_n_surf] = self.c_s_n_0
                self.model.boundary_conditions[self.c_s_n_surf] = {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(0), "Neumann"),}


    def define_output_variables(self):
        '''Output variables which are useful for plots'''
        voltage = pybamm.boundary_value(self.phi_s_p, "right")

        # calculate variables for overpotential contributions
        
        eta_elec = pybamm.x_average(self.phi_e_p)
        ohmic_losses = pybamm.boundary_value(self.phi_s_p, "right") - pybamm.x_average(self.phi_s_p)



        self.model.variables = {
            "Electrolyte concentration [mol.m-3]": self.c_e,
            "Separator electrolyte concentration [mol.m-3]": self.c_e_s, # TODO: remove, and make mod to pybamm
            "Positive electrolyte concentration [mol.m-3]": self.c_e_p, # TODO: remove, and make mod to pybamm
            "Separator electrolyte potential [V]": self.phi_e_s, # TODO: remove, and make mod to pybamm
            "Positive electrolyte potential [V]": self.phi_e_p,  # TODO: remove, and make mod to pybamm
            "Electrolyte potential [V]": self.phi_e,
            "Positive electrode potential [V]": self.phi_s_p,
            "Voltage [V]": voltage,
            "Time [s]": pybamm.t,
            "Positive electrode mean diffusion overpotential [V]": pybamm.Scalar(0),
            "Positive electrode mean electrolyte overpotential [V]": eta_elec,
            "Positive electrode mean ohmic losses [V]": ohmic_losses,
            "Positive electrode mean bulk equilibrium potential [V]": pybamm.Scalar(0),
            "j_p_ave": self.j_p_ave,
            "DC correction term positive electrode [mol.m-3]": self.correction_term_p,
        }

        if self.calc_soc:
            max_mass = self.vf_am_p * self.V_p * self.c_p_max
            total_mass = self.A_cs * self.vf_am_p * pybamm.Integral(self.c_s_p, self.x_p)
            soc_p = total_mass / max_mass
            self.model.variables.update({"Positive electrode SOC": soc_p,})          


        if self.cell_type == "Full cell":
            self.model.variables.update({"R-averaged negative particle concentration [mol.m-3]": self.c_s_n,
                                         "Negative particle surface concentration [mol.m-3]": self.c_s_n_surf,
                                         "Negative electrolyte concentration [mol.m-3]": self.c_e_n,
                                         "Negative electrolyte potential [V]": self.phi_e_n,
                                         "Negative electrode potential [V]": self.phi_s_n,
                                         "Negative electrode reaction overpotential [V]": self.eta_n,
                                         "Negative electrode interfacial current density [A.m-2]": self.j_n,
                                         "j_n_ave": self.j_n_ave,
                                         "DC correction term negative electrode [mol.m-3]": self.correction_term_n,
                                         })


        if self.sei:
            # calculate Q loss due to SEI growth
            if self.cell_type == "Full cell":
                A = self.av_n_real * self.V_n 
            else:
                A = self.av_p_real * self.V_p 

            self.model.variables.update({"SEI thickness [m]": self.L_sei, "SEI growth current density [A.m-2]": self.j_sei, "Surface area for Qloss calculation [m2]": A})
        
        if self.min_eta_plating:
            if self.cell_type == "Full cell":
                eta_plating = self.phi_s_n - self.phi_e_n
            else:
                if self.plating_correction_closure:
                    # ===== correction from b closure ======
                    phi_tilde = self.b_min * pybamm.grad(self.phi_e_p)
                    phi_e_min = self.phi_e_p + phi_tilde
                    eta_plating = self.phi_s_p - phi_e_min
                    eta_plating_min = pybamm.EvaluateAt(eta_plating, 0.0) # only take value at sep interface, as this is where b_min is defined
                    #========================================
                else:
                    # ==== No correction === 
                    eta_plating = self.phi_s_p - self.phi_e_p
                    eta_plating_min = pybamm.min(eta_plating)

                # adding variables for reconstruction of eta_plating
                grad_phi_e = pybamm.grad(self.phi_e_s)
                ones = pybamm.PrimaryBroadcastToEdges(pybamm.Scalar(1), "positive electrode")
                phi_e_faces = ones * self.phi_e_p
                phi_s_faces = ones * self.phi_s_p
                c_e_faces = ones * self.c_e_p
                j_faces = ones * self.j_p
                
                self.model.variables.update({"Minimum plating overpotential [V]": eta_plating_min, 
                                             "Grad phi_e [V.m-1]": grad_phi_e,
                                             "Electrolyte potential at faces [V]": phi_e_faces,
                                             "Solid potential at faces [V]": phi_s_faces,
                                             "Electrolyte concentration at faces [mol.m-3]": c_e_faces,
                                             "Reaction rate at faces [A.m-2]": j_faces})
                
        # ---- TODO: DELETE ME. For checking continuity BCs at interface ---- 
        grad_phi_e_p = pybamm.grad(self.phi_e_p)
        grad_phi_e_s = pybamm.grad(self.phi_e_s)
        ones_p = pybamm.PrimaryBroadcastToEdges(pybamm.Scalar(1), "positive electrode")
        ones_s = pybamm.PrimaryBroadcastToEdges(pybamm.Scalar(1), "separator")
        phi_e_faces_p = ones_p * self.phi_e_p
        phi_e_faces_s = ones_s * self.phi_e_s
        self.model.variables.update({
                                    "Grad phi_e_p [V.m-1]": grad_phi_e_p,
                                    "Grad phi_e_s [V.m-1]": grad_phi_e_s,
                                    "phi_e_p faces [V]": phi_e_faces_p,
                                    "phi_e_s faces [V]": phi_e_faces_s,
                                    })
        #====================================================================


    def define_geometry_and_discretise(self):
        # define geometry

        # add to output variables
        self.model.variables['x [m]'] = pybamm.concatenation(self.x_n, self.x_s, self.x_p)
        self.model.variables['x_n [m]'] = self.x_n
        self.model.variables['x_s [m]'] = self.x_s
        self.model.variables['x_p [m]'] = self.x_p

        geometry = {
            "negative electrode": {self.x_n: {"min": pybamm.Scalar(0), "max": self.L_n}},
            "separator": {self.x_s: {"min": self.L_n, "max": (self.L_n + self.L_s)}},
            "positive electrode": {self.x_p: {"min": (self.L_n + self.L_s), "max": (self.L_n + self.L_s + self.L_p)}},
        }

        self.param_values.process_model(self.model)
        self.param_values.process_geometry(geometry)

        # mesh and discretise
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
        }

        # define spatial methods
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume(),
            "separator": pybamm.FiniteVolume(),
            "positive electrode": pybamm.FiniteVolume(),
        }

        var_pts = {
            self.x_n: self.var_pts["x_n"],
            self.x_s: self.var_pts["x_s"],
            self.x_p: self.var_pts["x_p"],
        }

        self.mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        disc = pybamm.Discretisation(self.mesh, spatial_methods)
        disc.process_model(self.model)
        

    def j_bv(self, j0, eta, T):
        '''the butler volmer equation, returning rate in C.s-1.m-2'''
        alpha = 0.5
        return j0 * (pybamm.exp(alpha * self.F / (self.R * T) * eta)
                     - pybamm.exp(- (1 - alpha) * self.F / (self.R * T) * eta))

    
      

    def ramp_half_cosine(self, t, t0, dt, y_start, y_end):
        # ramps from y_start to y_end between t0 and t0 + dt using half cosine
        u = (t - t0) / dt
        y = (t > (t0 + dt)) * y_end + (t >= t0) * (t <= (t0 + dt)) * (y_start + 0.5 * (y_end - y_start) * (1 + np.cos(np.pi * u + np.pi)))
        return y

    def smooth_correction_term(self, j_p, s0_surface_average, prev_correction_term, dt=10.0):
        # smooths the correction term to avoid discontinuities at epoch switches
        y_end = j_p * s0_surface_average
        y_start = prev_correction_term["correction"]
        t0 = prev_correction_term["time"]

        return self.ramp_half_cosine(self.t, t0, dt, y_start, y_end)

    


    



        
        










