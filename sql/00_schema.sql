-- Création du schéma logique principal du projet Rakuten.
-- Ce fichier doit être exécuté une seule fois au démarrage du projet.

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;

CREATE SCHEMA IF NOT EXISTS project AUTHORIZATION mlops;
COMMENT ON SCHEMA project IS 'Schéma principal du projet Rakuten MLOps.';
