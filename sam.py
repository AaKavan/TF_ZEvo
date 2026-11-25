'''
Python file for running the semi-analytic model.
'''
from workbook_functions import *

def run_sam():
    global trees
    #Start from a fresh copy of the trees every time we rerun the model
    trees = np.load(TREE_FILE) 

    for i in range(0,len(scales)):
        scale = scales[i]
        redshift = 1./float(scale) - 1.
        dt = s_to_time(scale) #Time, in years, since the last snapshot
        if (i>0):
            dt -= s_to_time(scales[i-1])

        #Initialize cooling rate spline here!!!
        cooling_spline = mass_cooling_rate_spline(redshift)

        halo_indices = halos_at_scale[scale]

        #Convenience arrays
        mvir = trees['mvir'][halo_indices]
        mcold = trees['m_cold'][halo_indices]
        sm = trees['sm'][halo_indices]
        mejected = trees['m_ejected'][halo_indices]
        micl = trees['m_icl'][halo_indices]
        mmerged = trees['m_merged'][halo_indices]
        mbulge = trees['m_bulge'][halo_indices]

        #Initialize halos with no progenitors: their hot gas is f_b*M_halo, and nothing else exists yet
        no_prog = trees['num_prog'][halo_indices]==0
        mhot = trees['m_hot'][halo_indices]
        mhot = np.where(no_prog, mvir*fb, mhot) #Units of Msun/h
        mcold = np.where(no_prog, 0, mcold)
        sm = np.where(no_prog, 0, sm)
        mejected = np.where(no_prog, 0, mejected)
        micl = np.where(no_prog, 0, micl)
        mmerged = np.where(no_prog, 0, mmerged)
        mbulge = np.where(no_prog, 0, mbulge)

        #Calculate disk mass; note that SM = m_disk + m_merged + m_bulge
        m_disk = sm - mmerged - mbulge

        #Check if total incoming stellar mass is greater than the disk mass * r_bulge threshold
        #If so, the merging stars go into the bulge, along with an equal amount of the disk
        # (or the whole disk, if it is smaller than the amount of merging stars)
        # major_merger = ...  #You fill this in! (True for halos with a large enough merger)
        # bulge_growth = ...  #You fill this in! (Msun/h; use np.where or np.minimum)
        
        ### classify as major merger if mass is greater than r_bulge * m_disk, which is our threshold
        major_merger = mmerged > r_bulge * m_disk ### Boolean
        
        ### all the satellites go into the bulge, plus an equal mass of disk stars
        ### or the whole disk, if the disk is smaller than that
        bulge_growth = mmerged + np.minimum(mmerged, m_disk)

        ### adds bulge growth only for when major merger is true
        mbulge = np.where(major_merger, mbulge+bulge_growth, mbulge)

        #Calculate cooling rates from mvir and mhot for every halo at this scale factor at once
        cooling_rate = fast_mass_cooling_rate(cooling_spline,mvir,mhot) #in Msun/h / yr

        #Calculate mass loading factors
        #float64 matters here: older numpy (<2.0) would return float32, and the unit conversions below overflow
        t_vir = np.float64(fast_T_vir(redshift, mvir))                   #in K
        f_sn = supernovae_reheating_ratio(t_vir)                         #Reheating mass loading factor
        f_eject = supernovae_ejection_ratio(trees['vmax'][halo_indices]) #Ejection mass loading factor
        f_net = f_sn + f_eject                                           #Total mass-loading factor

        #Now calculate black hole reheating rates
        #First, need the reheating temperature in Kelvin: the virial temperature or 10^6 K, whichever is lower
        t_reheat = np.minimum(t_vir, 1e6)

        #Try the fast mass accretion rate solution (power ~ 0.01 Eddington)
        mbh = alpha*sm #We assume that M_bh is proportional to stellar mass; Msun/h
        edd_rate = eddington_rate(mbh) #Msun/h/yr
        fast_heating_rate = bh_fast_mass_reheating_rate(mbh, t_reheat) #Msun/h/yr
        bh_acc_rate = alpha*(cooling_rate - fast_heating_rate)/(1.+f_net) #BH accretion rate for the fast-accretion case; Msun/h/yr
        net_cooling_rate = cooling_rate - fast_heating_rate #Cooling rate available for forming stars, Msun/h/yr

        #Check if this solution works--if not, then use the slow mass accretion rate solution (power ~ acc rate)
        slow_accretion = (bh_acc_rate < 0.01*edd_rate)
        f_net = np.where(slow_accretion, f_net + bh_slow_mass_loading_factor(t_reheat), f_net)
        net_cooling_rate = np.where(slow_accretion, cooling_rate, net_cooling_rate)

        #Calculate SFR for the steady-state solution
        sfr = net_cooling_rate/(1.+f_net) #in Msun/h / yr

        #Calculate the cold gas mass necessary to sustain that SFR
        trees['re_gas'][halo_indices] = re_gas(trees['rvir'][halo_indices]) #comoving kpc/h
        m_cold_needed = KS_Cold_Gas(sfr,trees['re_gas'][halo_indices],scale,ks_constant) #in Msun/h
        amount_to_cool = m_cold_needed-mcold

        #Calculate the time needed to cool that much gas.  Halos with no hot gas can't cool at all,
        #so they get the whole snapshot (during which nothing cools anyway).
        nonzero_rate = np.where(cooling_rate>0, cooling_rate, 1.) #Avoids dividing by zero below
        t_cool = np.where(cooling_rate>0, amount_to_cool/nonzero_rate, dt) #In yr
        t_cool = np.minimum(t_cool, dt) #Can't cool for longer than the snapshot lasts

        #Case 1: we need to cool more gas, so assume SFR=0 until enough cold gas is present
        cooled_gas = np.minimum(t_cool*cooling_rate, mhot) #Msun/h; don't cool more gas than the halo has
        #Form stars at a constant rate for the rest of the snapshot, or until the hot gas runs out:
        stars_from_cooling = np.minimum(sfr*(dt-t_cool), (mhot-cooled_gas)/(1.+f_eject)) #Msun/h

        #Case 2: there's already too much cold gas, so burn the excess in an immediate starburst
        amount_to_heat = -amount_to_cool       #Msun/h
        mhot_after_burst = mhot+amount_to_heat #Part of this becomes stars + ejected gas below
        stars_from_burst = amount_to_heat/(1.+f_net) #Stars made in the initial starburst, in Msun/h
        #...after which stars keep forming at the rate above, until the hot gas runs out.
        #The starburst has already used up some of it, so measure what is actually left:
        mhot_left = mhot_after_burst-stars_from_burst*(1.+f_eject)
        stars_from_burst = stars_from_burst+np.minimum(sfr*dt, mhot_left/(1.+f_eject))

        #Pick whichever case applies to each halo (t_cool<=0 means we already have enough cold gas)
        need_to_cool = (t_cool>0)
        mcold = np.where(need_to_cool, mcold+cooled_gas, mcold-amount_to_heat) #Mass conservation
        mhot = np.where(need_to_cool, mhot-cooled_gas, mhot_after_burst)
        new_stars = np.where(need_to_cool, stars_from_cooling, stars_from_burst) #Msun/h

        #For both cases: remove the hot gas that went into stars and into the wind
        mhot = mhot-new_stars*(1.+f_eject)    #Msun/h
        mejected = mejected+new_stars*f_eject #Msun/h
        sm = sm+new_stars
        mbh = mbh+alpha*new_stars #Note that M_bh is proportional to alpha*SM

        #Part 3: add your disk stability check here!  Any disk mass above the maximum stable disk mass
        # (for a disk radius R_disk = re_stars, in physical kpc/h) should be transferred into the bulge.
        
        ### Let's set disk instability to false, and only set it to true when running part 3
        if disk_instability == True:
            ### it's comoving, so convert it to physical kpc/h
            ### Q = V_max / sqrt(GM/R) >= 1
            ### and Q = 1 => M_disk_max = V_max^2 R/G
            ### 1 percent of r virial
            r_disk = 0.01 * trees['rvir'][halo_indices] * scale
            m_disk_max = inverse_G * trees['vmax'][halo_indices]**2 * r_disk
            ### sm = mdisk + mbulge
            m_disk = sm - mbulge
            ### update the bulge mass
            ### DONT add a negative mass
            mbulge = mbulge + np.maximum(m_disk - m_disk_max, 0)

            
        

        #Always write the temporary arrays back into the tree!
        trees['sfr'][halo_indices] = new_stars/dt #Average star formation rate, Msun/h/yr
        trees['m_hot'][halo_indices] = mhot
        trees['m_cold'][halo_indices] = mcold
        trees['sm'][halo_indices] = sm
        trees['m_ejected'][halo_indices] = mejected
        trees['m_bh'][halo_indices] = mbh
        trees['m_icl'][halo_indices] = micl
        trees['m_bulge'][halo_indices] = mbulge

        #Transfer gas, stars, and black holes from most-massive progenitors to their descendant halos
        all_desc_indices = trees['desc_index'][halo_indices]
        mmp_and_desc_exists = (trees['mmp?'][halo_indices]==1) & (all_desc_indices>-1)
        mmp_desc_indices = all_desc_indices[mmp_and_desc_exists]
        mass_differences = trees['mvir'][mmp_desc_indices] - mvir[mmp_and_desc_exists]
        trees['m_cold'][mmp_desc_indices] = mcold[mmp_and_desc_exists]
        trees['m_ejected'][mmp_desc_indices] = mejected[mmp_and_desc_exists]
        trees['sm'][mmp_desc_indices] += sm[mmp_and_desc_exists]  #Note that we do "+=" instead of "=" now!
        trees['m_bh'][mmp_desc_indices] += mbh[mmp_and_desc_exists]
        trees['m_icl'][mmp_desc_indices] += micl[mmp_and_desc_exists]
        trees['m_bulge'][mmp_desc_indices] = mbulge[mmp_and_desc_exists]
        #Add the halo's mass growth x the cosmic baryon fraction to the descendant's hot gas
        trees['m_hot'][mmp_desc_indices] = np.maximum(mhot[mmp_and_desc_exists]+fb*mass_differences, 0)

        #Satellites merging into their descendant halo: their stars and black holes go to the descendant's
        # central galaxy, except for a fraction f_ICL of the stars, which is disrupted into the descendant's ICL.
        # The satellite's own ICL always ends up in the descendant's ICL.
        sat_and_desc_exists = (trees['mmp?'][halo_indices]!=1) & (all_desc_indices>-1)
        sat_desc_indices = all_desc_indices[sat_and_desc_exists]
        #Several satellites can merge into the same descendant, so we use np.add.at() instead of "+=":
        # trees['sm'][sat_desc_indices] += ... would only add in one of them!
        np.add.at(trees['sm'], sat_desc_indices, (1-f_ICL)*sm[sat_and_desc_exists])
        np.add.at(trees['m_bh'], sat_desc_indices, (1-f_ICL)*mbh[sat_and_desc_exists])
        np.add.at(trees['m_icl'], sat_desc_indices, f_ICL*sm[sat_and_desc_exists]+micl[sat_and_desc_exists])
        #Also keep a separate tally of the merging stars, for the bulge-formation check above
        # np.add.at(trees['m_merged'], sat_desc_indices, ...) #You fill this in!
        np.add.at(trees['m_merged'], sat_desc_indices, (1-f_ICL)*sm[sat_and_desc_exists])
