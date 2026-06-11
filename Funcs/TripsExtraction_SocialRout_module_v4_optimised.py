#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8th Novemeber 2023:

author: Alex Roocroft

"""

# Extraction of simulated vehicles from symuvia xml output

import xml.sax
import pandas as pd
import numpy as np
import csv
import gc
from lxml import etree

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
    #-----------------------------------
    # parser = xml.sax.make_parser()
    # parser.setContentHandler(ParserHandler(sim_hours))
    # parser.parse(open(temp_folder + symuvia_output, "r"))

    # -------------memory reducing patch -------0----------

    tree = etree.parse(temp_folder + symuvia_output)
    root = tree.getroot()

    # manually traverse VEH objects
    vehs = []
    links = []

    for elem in root.iter():
        tag = elem.tag
        if tag == "TRONCON":
            if 'id_eltaval' in elem.attrib:
                links.append({
                    'id': elem.attrib['id'],
                    'dowstreamnode': elem.attrib['id_eltaval']
                })

        elif tag == "VEH":
            attrs = elem.attrib
            creation = float(attrs['instC'])
            exit_time = float(attrs.get('instS', -1))
            travel = exit_time - creation if exit_time > 0 else (3600 * sim_hours) - creation

            vehs.append({
                'id': int(attrs['id']),
                'type': attrs['type'],
                'origin': attrs['entree'],
                'destination': attrs.get('sortie', ''),
                'creation_time': creation,
                'exit_time': exit_time,
                'travel_time': travel,
                'path': attrs.get('itineraire', '')
            })


    #----------------------------------------

    # Compute incomplete trips WITHOUT DataFrame
    incomplete_trips = sum(
        1 for v in vehs
        if v["type"] == "VL" and v["exit_time"] < 0
    )
    # if incomplete_trips > 0:
    #     print(f"[Cycle {iteration_cycle}][Day {iteration_day}] Extraction... # of not terminated vehicles = {incomplete_trips}")

    # Keep only VL vehicles
    vehs_vl = [v for v in vehs if v["type"] == "VL"]

    # Fast mapping for link → downstreamnode
    link_map = {link["id"]: link["dowstreamnode"] for link in links}

    # Assign missing destinations
    for v in vehs_vl:
        if v["destination"] == "xxx":
            lastlink = v["path"].split()[-1]
            v["destination"] = link_map.get(lastlink, v["destination"])

    # Compute network statistics
    total_time_completed = sum(
        v["travel_time"] for v in vehs_vl if v["exit_time"] > 0
    )

    network_stats = {
        "total_time": total_time_completed,
        "total_time_completed": total_time_completed
    }

    # Save extracted vehicle data to CSV
    with open(out_folder + f'vehs_Day_{iteration_day}_cycle_{iteration_cycle}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'type', 'origin', 'destination', 'creation_time',
                                                     'entry_time', 'exit_time', 'travel_distance', 'travel_time',
                                                     'path'], delimiter=';')
        writer.writeheader()
        for v in vehs_vl:
            writer.writerow(v)

    # Direct NumPy conversion
    driver_tt_np = np.array(
        [v["travel_time"] for v in vehs_vl],
        dtype=np.float32
    )

    veh_output = {
        'network_stats': network_stats,
        'driver_travel_time': driver_tt_np,
        'incomplete_trips': incomplete_trips
    }

    return veh_output
