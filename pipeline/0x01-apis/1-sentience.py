#!/usr/bin/env python3
""" Sentience """
import requests


def sentientPlanets():
    """ returns the list of names of the home planets of all
        sentient species """
    total_planets = []
    next = "https://swapi-api.hbtn.io/api/species/"
    while next:
        r = requests.get(next)
        if r.status_code != 200:
            break
        data = r.json()
        next = data['next']
        people = data['results']
        for person in people:
            if person['designation'] == 'sentient' or\
               person['classification'] == 'sentient':
                if person['homeworld']:
                    r = requests.get(person['homeworld'])
                    if r.status_code == 200:
                        planet = r.json()
                        total_planets.append(planet['name'])
    return total_planets
