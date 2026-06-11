#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8th Novemeber 2023:

author: Alex Roocroft

"""

# Extraction of simulated vehicles from symuvia xml output

import xml.sax
import pandas as pd

import csv


class ParserHandler(xml.sax.ContentHandler):
    """
    SAX parser handler to extract vehicle and link data from a SymuVia XML output file.
    """

    def __init__(self, sim_hours):
        self.sim_hours = sim_hours
        global vehs
        global links
        vehs = []  # List to store vehicle data
        links = []  # List to store link data

    def startElement(self, name, attrs):
        """
        Called when an XML element is encountered.
        Extracts relevant vehicle and link data.
        """
        if name == "TRONCON":
            if 'id_eltaval' in attrs:
                link = {'id': attrs['id'], 'dowstreamnode': attrs['id_eltaval']}
                links.append(link)

        if name == "VEH":
            veh = {
                'id': int(attrs['id']),
                'type': attrs['type'],
                'origin': attrs['entree'],
                'destination': attrs.get('sortie', ''),
                'creation_time': float(attrs['instC']),
                'entry_time': float(attrs.get('instE', attrs['instC'])),
                'exit_time': float(attrs.get('instS', -1.0)),
                'travel_distance': float(attrs.get('dstParcourue', 0.0)),
                'path': attrs.get('itineraire', '')
            }

            # Compute travel time
            if veh['exit_time'] > 0:
                veh['travel_time'] = veh['exit_time'] - veh['creation_time']
            else:
                veh['travel_time'] = (3600 * self.sim_hours) - veh['creation_time']

            vehs.append(veh)


def symmvia_output(out_folder, temp_folder, symuvia_output, iteration_day, iteration_cycle, sim_hours):
    """
    Parses the SymuVia XML output file and extracts vehicle travel data.

    Parameters:
        out_folder (str): Directory for saving extracted vehicle data.
        temp_folder (str): Directory containing the SymuVia output file.
        symuvia_output (str): Filename of the XML output file.
        iteration_day (int): Simulation day index.
        iteration_cycle (int): Simulation cycle index.
        sim_hours (int): Number of simulated hours.

    Returns:
        dict: Extracted travel statistics including network stats, travel times, and incomplete trips.
    """
    parser = xml.sax.make_parser()
    parser.setContentHandler(ParserHandler(sim_hours))
    parser.parse(open(temp_folder + symuvia_output, "r"))

    global vehs, links
    df_vehs = pd.DataFrame(vehs)
    df_links = pd.DataFrame(links)

    # Print summary of vehicle types
    for vehicle_type in ['VL', 'PL', 'BUS', 'TRAM', 'METRO']:
        print(vehicle_type, len(df_vehs[df_vehs['type'] == vehicle_type]))

    # Count the number of incomplete trips (vehicles that never reached a destination)
    incomplete_trips = len(df_vehs.loc[(df_vehs['type'] == 'VL') & (df_vehs['exit_time'] < 0)])
    print('# of not terminated vehicles', incomplete_trips)

    # Filter only vehicle trips
    vehs = [veh for veh in vehs if veh['type'] == 'VL']

    # Assign missing destinations based on last link in path
    for veh in vehs:
        if veh['destination'] == 'xxx':
            lastlink = veh['path'][veh['path'].rfind(' ') + 1:]
            veh['destination'] = df_links.loc[df_links['id'] == lastlink, 'dowstreamnode'].values[0]

    # Compute network statistics
    network_stats = {
        'total_time': df_vehs.loc[(df_vehs['type'] == 'VL') & (df_vehs['exit_time'] > 0), 'travel_time'].sum(),
        'total_time_completed': df_vehs.loc[(df_vehs['type'] == 'VL') & (df_vehs['exit_time'] > 0), 'travel_time'].sum()
    }

    # Save extracted vehicle data to CSV
    with open(out_folder + f'vehs_Day_{iteration_day}_cycle_{iteration_cycle}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'type', 'origin', 'destination', 'creation_time',
                                                     'entry_time', 'exit_time', 'travel_distance', 'travel_time',
                                                     'path'], delimiter=';')
        writer.writeheader()
        for veh in vehs:
            writer.writerow(veh)

    veh_pd = pd.DataFrame(vehs)
    veh_output = {
        'network_stats': network_stats,
        'driver_travel_time': veh_pd['travel_time'],
        'incomplete_trips': incomplete_trips
    }

    return veh_output
