import numpy as np
import pandas as pd
import flopy
import flopy.utils.binaryfile as bf
import rasterio

# Load necessary input for hydrologic modeling
# DEM
DEM  = rasterio.open('../data/characterization_data/UAS/DEM_RTK.tif').read(1)
no_DEM_mask = DEM<0
DEM[DEM<0] = 2740. # Give a default value for no DEM area but wouldn't change the actual simulation
bounds = rasterio.open('../data/characterization_data/UAS/DEM_RTK.tif').bounds
xMin = bounds[0]
yMin = bounds[1]

# Load structure
gravel_interface = np.load('../model/subsurface/predict_gravel.npy')
gravel_interface[np.isnan(gravel_interface)] = 0
bedrock_interface = np.load('../model/subsurface/predict_bedrock.npy')
bedrock_interface[np.isnan(bedrock_interface)] = 0

# River
river = np.load('../data/characterization_data/Beaver_pond_dam/river.npy')
river_flow_area = np.load('../data/characterization_data/Beaver_pond_dam/river_flow_area.npy')

# Dam
dam = np.load('../data/characterization_data/Beaver_pond_dam/dam.npy')

# Pond
pond = np.load('../data/characterization_data/Beaver_pond_dam/pond_baseflow.npy')

# Period forcings
forcings = pd.read_csv('../data/response_data/preprocessing/period_forcings.csv')

def modflow_BC(hk_gravel,
               hk_soil,
               vka_ratio_gravel,
               vka_ratio_soil,
               k_dam,
               ET,
               Precip,
               model_name, 
               model_dir,
               period = 'baseflow',
               structure_ratio = [1,1]):
    
    if period == 'baseflow':
        pond = np.load('../data/characterization_data/Beaver_pond_dam/pond_baseflow.npy')
    elif period == 'snowmelt':
        pond = np.load('../data/characterization_data/Beaver_pond_dam/pond_snowmelt.npy')
    elif period == 'dry':
        pond = np.load('../data/characterization_data/Beaver_pond_dam/pond_baseflow.npy')
        pond[:] = 0
        
    pond_level,river_diff = np.array(forcings[forcings['Period']==period].values[0][3:],dtype = 'float64')
    

    
    # Other inputs: 
    # DEM, dam, river_flow_area, river, pond, no_DEM_mask, gravel_interface, bedrock_interface
    
    # model size and res
    Lx = 480.
    Ly = 460.
    ncol = 480
    nrow = 460
    
    delr = Lx/ncol # spacings along a row, can be an array
    delc = Ly/nrow # spacings along a column, can be an array
    
    nlay = 16
    soil_nlay = 10
    gravel_nlay = nlay - soil_nlay
    dam_nlay = 5
    
    # Build different layers from our reconstructed floodplain structure
    mf = flopy.modflow.Modflow(modelname = model_name, model_ws = model_dir, exe_name='mfnwt')

    # setting up the vertical discretization and model bottom elevation
    zbot = np.zeros((nlay,nrow,ncol))
    
    # Soil layers
    for lay in np.arange(0,soil_nlay):    
        zbot[lay,:,:] = DEM - np.maximum(gravel_interface*structure_ratio[0]*((lay+1)/soil_nlay),0.1*(lay+1)) 
    
    # Gravel layers
    gravel_discretized_ratio = [0.02,0.04,0.1,0.3,0.6,1]
    for i, lay in enumerate(np.arange(soil_nlay, nlay)):
        zbot[lay,:,:] = zbot[soil_nlay-1,:,:] - np.maximum(bedrock_interface*structure_ratio[1]*gravel_discretized_ratio[i],0.1*(i+1))
     
    # Dis package 
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay,
                                   nrow=nrow, ncol=ncol,
                                   delr=delr, delc=delc,
                                   top=DEM, botm=zbot,
                                   itmuni=1.,nper=1,perlen=1,nstp=1,steady=True)

    mf.modelgrid.set_coord_info(xoff=xMin, yoff=yMin, angrot=0, epsg=26913)
    
    # Variables for the BAS package
    # ibound:   active > 0, inactive = 0, or constant head < 0
    ## Let's deactivate places we don't have DEM
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:,no_DEM_mask] = 0
    
    ## Let's put constant head at river and pond
    ibound[0,:,:][river_flow_area==1] = -1
    ibound[0,:,:][pond==1] = -1
    
    ## At the DAM we don't need to have a constant head
    ibound[0,:,:][dam==1] = 1 
    
    # Boundary condition
    strt = np.zeros((nlay, nrow, ncol), dtype=np.float32) + 2740.
    ## River: constant head
    strt[0,:,:][river_flow_area==1] = DEM[river_flow_area==1]+river_diff
    ## Pond: constant head
    strt[0,:,:][pond==1] = pond_level
    
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

    # ET
    evt = flopy.modflow.ModflowEvt(mf, nevtop=3, evtr=ET)
    
    # Rch (Recharge only) 
    # Recharge is not the same as precipitation. 
    # Recharge includes only that portion of the precipitation that actually reaches the water table. 
    # Evapotranspiration from the unsaturated zone is not included in recharge. 
    # Recharge may also include flow into groundwater from streams or rivers if that flow is not included in the model in some other way.
    RCH = Precip-ET
    rch = flopy.modflow.ModflowRch(mf, nrchop=3, rech=RCH)
    
    # hydraulic conductivity, horizontal
    # input: m/s *3600*24 -> m/d
    hk = np.zeros((nlay, nrow, ncol))
    hk = hk + (np.array([hk_soil]*soil_nlay+[hk_gravel]*gravel_nlay)*3600*24).reshape(nlay,1,1)
    
    ## Riverbed
    for lay in range(soil_nlay):    
         hk[lay,np.where(river==1)[0],np.where(river==1)[1]] = hk_gravel*3600*24
    
    ## DAM   
    dam_location = np.where(dam==1)
    for lay in range(dam_nlay): # top 5 layers with beaver dam
        hk[lay,dam_location[0],dam_location[1]] =  k_dam*3600*24
    
    # hydraulic conductivity, vertical
    vka_ratio = np.zeros((nlay, nrow, ncol))
    vka_ratio = vka_ratio + (np.array([vka_ratio_soil]*soil_nlay+[vka_ratio_gravel]*gravel_nlay)).reshape(nlay,1,1)
    
    ## Gravel river
    for lay in range(soil_nlay): 
        vka_ratio[lay,np.where(river==1)[0],np.where(river==1)[1]] = vka_ratio_gravel
    
    vka = hk*vka_ratio
    
    ## DAM, vka
    for lay in range(dam_nlay): 
        vka[lay,dam_location[0],dam_location[1]] =  k_dam*3600*24

    laytyp = np.zeros(nlay)
    
    # Dam sediment wedge
    for i in range(5): # 5m wide
        hk[0:dam_nlay,dam_location[0],dam_location[1]-i-1] = k_dam*3600*24
        vka[0:dam_nlay,dam_location[0],dam_location[1]-i-1] = k_dam*3600*24

    lpf = flopy.modflow.ModflowLpf(mf, hk=hk,vka=vka,laytyp = laytyp, ipakcb=53)

    spd = {(0, 0): ['print head', 'print budget','save head', 'save budget']}
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

    # Add PCG package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(mf)

    # Write the MODFLOW model input files
    mf.write_input()

    # Run the MODFLOW model
    success, buff = mf.run_model()

    if success: 
        hds = bf.HeadFile(model_dir+'/'+model_name+'.hds')
        times = hds.get_times() # simulation time, steady state
        head = hds.get_data(totim=times[-1])
        cbb = bf.CellBudgetFile(model_dir+'/'+model_name+'.cbc') # read budget file
        flf = cbb.get_data(text='FLOW LOWER FACE', totim=times[-1])[0]
        frf = cbb.get_data(text='FLOW RIGHT FACE', totim=times[-1])[0]
    else: 
        head = np.nan
        flf = np.nan
        frf = np.nan
    return mf,head,hk,vka,strt,zbot,flf,frf


def read_sim(model_dir, model_name):
    hds = bf.HeadFile(model_dir+'/'+model_name+'.hds')
    times = hds.get_times() # simulation time, steady state
    head = hds.get_data(totim=times[-1])
    cbb = bf.CellBudgetFile(model_dir+'/'+model_name+'.cbc') # read budget file
    flf = cbb.get_data(text='FLOW LOWER FACE', totim=times[-1])[0]
    frf = cbb.get_data(text='FLOW RIGHT FACE', totim=times[-1])[0]
    
    return head, flf, frf